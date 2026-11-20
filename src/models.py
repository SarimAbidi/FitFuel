from abc import ABC, abstractmethod
import math
import numpy as np

from .catalogue import eligible_foods, reviewed_foods
from .config import (ACTIVITY, GOALS, MAX_PLAN_ITEMS, MEALS, MIN_ESTIMATE_CALORIES, MIN_LOSS_BMI,
                     NUTRIENTS, PREFERENCES, WEIGHT_RANGE)


class InvalidProfileError(ValueError):
    """Raised when a profile cannot be used by this educational planner."""


class UserProfile:

    def __init__(self, age, weight, height, sex, activity, goal,
                 diet="non-vegetarian", preference="balanced", target_weight=None):
        values = {"age": age, "weight": weight, "height": height}
        limits = {"age": (19, 78), "weight": WEIGHT_RANGE, "height": (130, 220)}
        if target_weight is not None:
            values["target_weight"] = target_weight
            limits["target_weight"] = WEIGHT_RANGE
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidProfileError(f"{key.replace('_', ' ')} must be a number.")
            low, high = limits[key]
            if not math.isfinite(value) or not low <= value <= high:
                raise InvalidProfileError(f"{key.replace('_', ' ')} must be between {low} and {high}.")
        if int(age) != age:
            raise InvalidProfileError("Age must be a whole number.")
        choices = [(sex, ("male", "female"), "formula sex"),
                   (activity, ACTIVITY, "activity"), (goal, GOALS, "goal"),
                   (diet, ("vegetarian", "non-vegetarian"), "diet"),
                   (preference, PREFERENCES, "preference")]
        for value, allowed, field in choices:
            if not isinstance(value, str) or value not in allowed:
                raise InvalidProfileError(f"Invalid {field}.")
        minimum = MIN_LOSS_BMI * (height / 100) ** 2
        if goal == "lose" and weight < minimum:
            raise InvalidProfileError(f"Weight-loss estimates are disabled below BMI {MIN_LOSS_BMI}.")
        if target_weight is not None:
            if goal == "lose" and not minimum <= target_weight < weight:
                raise InvalidProfileError("A weight-loss target must be below your weight and within the supported range.")
            if goal == "gain" and target_weight <= weight:
                raise InvalidProfileError("A weight-gain target must be above your current weight.")
            if goal == "maintain" and target_weight != weight:
                raise InvalidProfileError("A maintenance target must match your current weight.")
        self._data = dict(values, sex=sex, activity=activity, goal=goal,
                          diet=diet, preference=preference)

    @property
    def data(self):
        """Return a copy so callers cannot bypass validation."""
        return self._data.copy()

    @property
    def goal_factor(self):
        p = self._data
        factor = GOALS[p["goal"]]
        if "target_weight" not in p:
            return factor
        change = (p["target_weight"] - p["weight"]) / p["weight"]
        return min(GOALS["gain"], max(GOALS["lose"], 1 + change))

    def targets(self):
        """Use Mifflin-St Jeor plus explicit, approximate project rules."""
        p = self._data
        offset = 5 if p["sex"] == "male" else -161
        resting = 10 * p["weight"] + 6.25 * p["height"] - 5 * p["age"] + offset
        maintenance = resting * ACTIVITY[p["activity"]]
        calories = max(resting, maintenance * self.goal_factor)
        if calories < MIN_ESTIMATE_CALORIES:
            raise InvalidProfileError("Estimate is outside this project's supported range.")
        return {"calories": calories, "protein": calories * 0.25 / 4,
                "carbs": calories * 0.45 / 4, "fat": calories * 0.30 / 9,
                "resting": resting, "maintenance": maintenance}


class Recommender(ABC):

    @abstractmethod
    def recommend(self, foods, profile, category=None, limit=8):
        """Return foods ranked for a profile."""


class GoalBasedRecommender(Recommender):

    def recommend(self, foods, profile, category=None, limit=8):
        p = profile.data
        chosen = eligible_foods(reviewed_foods(foods), profile).copy()
        if category:
            chosen = chosen[chosen["category"] == category]
        if chosen.empty:
            return chosen
        matrix = chosen[list(NUTRIENTS)].to_numpy(dtype=float)
        normalized = matrix / np.maximum(matrix.max(axis=0), 1)
        kcal, protein, _, _, fibre, sugar = normalized.T
        score = protein + fibre - 0.5 * np.abs(kcal - 0.5)
        goal_score = {"lose": protein + fibre - kcal, "gain": protein + kcal}
        if p["goal"] in goal_score:
            strength = abs(profile.goal_factor - 1) / abs(GOALS[p["goal"]] - 1)
            score += strength * (goal_score[p["goal"]] - score)
        if p["preference"] == "high-protein":
            score += protein
        elif p["preference"] == "high-fibre":
            score += fibre
        elif p["preference"] == "lower-sugar":
            score -= sugar
        chosen["score"] = score
        rows = sorted(chosen.to_dict("records"), key=lambda food: food["score"], reverse=True)
        return chosen.set_index("food_id").loc[
            [food["food_id"] for food in rows[:limit]]].reset_index()


class MealPlan:
    """Store food IDs and gram amounts; always calculate from the dataset."""

    def __init__(self):
        self._items = []

    @property
    def items(self):
        return [item.copy() for item in self._items]

    def copy(self):
        plan = MealPlan()
        plan._items = self.items
        return plan

    def add(self, food_id, grams, meal, foods):
        if len(self._items) >= MAX_PLAN_ITEMS:
            raise ValueError(f"A plan can contain up to {MAX_PLAN_ITEMS} entries.")
        if not isinstance(food_id, str) or food_id not in set(foods["food_id"]):
            raise ValueError("Food ID is not in the planner catalogue.")
        if meal not in MEALS:
            raise ValueError("Unknown meal name.")
        self._items.append({"food_id": food_id, "grams": self._validate_grams(grams), "meal": meal})

    def remove(self, number):
        self._items.pop(self._entry_index(number))

    def change_portion(self, number, grams):
        self._items[self._entry_index(number)]["grams"] = self._validate_grams(grams)

    def _entry_index(self, number):
        if not isinstance(number, int) or not 1 <= number <= len(self._items):
            raise ValueError("Invalid entry number.")
        return number - 1

    @staticmethod
    def _validate_grams(grams):
        if isinstance(grams, bool) or not isinstance(grams, (int, float)):
            raise ValueError("Grams must be a number.")
        if not math.isfinite(grams) or not 1 <= grams <= 1500:
            raise ValueError("Enter a portion between 1 and 1500 grams.")
        return float(grams)

    def totals(self, foods, meal=None):
        if not self._items:
            return np.zeros(len(NUTRIENTS))
        ids = [item["food_id"] for item in self._items]
        matrix = foods.set_index("food_id").loc[ids, list(NUTRIENTS)].to_numpy(dtype=float)
        portions = np.array([item["grams"] for item in self._items])[:, None] / 100
        scaled = matrix * portions
        if meal is not None:
            mask = np.array([item["meal"] == meal for item in self._items])
            scaled = scaled[mask]
        return scaled.sum(axis=0)
