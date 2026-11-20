from math import isclose, prod

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .catalogue import eligible_foods, reviewed_foods
from .config import MEALS, MIN_ESTIMATE_CALORIES, NUTRIENTS
from .models import MealPlan


NUTRIENT_SCALE = np.array([100, 20, 25, 12])
MEAL_SHARES = (0.25, 0.30, 0.30, 0.15)
MAX_MEAL_EXAMPLES = 150_000


def _portions(foods, grams):
    amounts = np.tile(grams, len(foods))
    ids = np.repeat(foods["food_id"].to_numpy(), len(grams))
    nutrients = np.repeat(foods[list(NUTRIENTS)].to_numpy(dtype=float), len(grams), axis=0)
    return ids, amounts, nutrients * amounts[:, None] / 100


class _MealOptions:
    def __init__(self, *roles):
        sizes = [len(ids) for ids, _, _ in roles]
        if not all(sizes):
            raise ValueError("The allowed catalogue is missing a food group needed for a starter plan.")
        combinations = prod(sizes)
        if combinations <= MAX_MEAL_EXAMPLES:
            examples = np.arange(combinations)
        else:
            examples = np.random.default_rng(42).choice(combinations, MAX_MEAL_EXAMPLES, replace=False)
        choices = np.unravel_index(examples, sizes)
        self.ids = np.column_stack([ids[index] for (ids, _, _), index in zip(roles, choices)])
        self.grams = np.column_stack([grams[index] for (_, grams, _), index in zip(roles, choices)])
        self.nutrients = sum(nutrients[index] for (_, _, nutrients), index in zip(roles, choices))
        self.model = NearestNeighbors().fit(self.nutrients[:, :4] / NUTRIENT_SCALE)

    def closest(self, target, used_foods, preference, minimum_calories=0):
        query = target[None, :] / NUTRIENT_SCALE
        distances, indices = self.model.kneighbors(query, n_neighbors=min(48, len(self.ids)))
        neighbors = indices[0]
        distance = distances[0]
        eligible = self.nutrients[neighbors, 0] >= minimum_calories
        if not eligible.any():
            neighbors = np.flatnonzero(self.nutrients[:, 0] >= minimum_calories)
            if not len(neighbors):
                raise ValueError("The catalogue could not assemble a starter plan within the supported intake range.")
            distance = np.linalg.norm(self.nutrients[neighbors, :4] / NUTRIENT_SCALE - query, axis=1)
        else:
            neighbors, distance = neighbors[eligible], distance[eligible]

        repeats = np.isin(self.ids[neighbors], list(used_foods)).sum(axis=1)
        score = distance + 0.10 * repeats
        if preference == "high-protein":
            score -= 0.005 * self.nutrients[neighbors, 1]
        elif preference == "high-fibre":
            score -= 0.02 * self.nutrients[neighbors, 4]
        elif preference == "lower-sugar":
            score += 0.005 * self.nutrients[neighbors, 5]
        return neighbors[score.argmin()]


def _legacy_roles(foods):
    """Use the original meal roles for catalogues without a starter_role column."""
    ids, category = foods["food_id"], foods["category"]
    oats = ids.eq("08121")
    dates = ids.eq("09421")
    peanut_butter = ids.eq("16098")
    starchy_vegetables = ids.isin(("11168", "11305", "11365", "11508"))
    dairy = category.eq("Dairy and eggs") & ~ids.eq("01129")
    fruit = category.eq("Fruit") & ~dates
    fats = category.eq("Nuts and seeds") | peanut_butter
    protein = category.isin(("Poultry", "Legumes")) & ~peanut_butter
    starch = (category.eq("Grains and cereals") & ~oats) | starchy_vegetables
    vegetables = category.eq("Vegetables") & ~starchy_vegetables
    return np.select([oats, dairy, fruit, fats, protein, starch, vegetables],
                     ["breakfast_grain", "dairy", "fruit", "fat", "protein", "starch", "vegetable"],
                     default="manual")


def _fit_meals(foods):
    roles = foods["starter_role"] if "starter_role" in foods else _legacy_roles(foods)

    def portions(role, grams):
        return _portions(foods[roles == role], grams)

    fruit_portions = portions("fruit", (100, 150, 200))
    fat_portions = portions("fat", (10, 20, 30, 40))
    breakfast = _MealOptions(portions("breakfast_grain", (100, 200, 300, 400)),
                             portions("dairy", (100, 175, 250, 325)),
                             fruit_portions, fat_portions)
    main = _MealOptions(portions("protein", (100, 150, 200, 250)),
                        portions("starch", (100, 175, 250, 325)),
                        portions("vegetable", (100, 175, 250)), fat_portions)
    snack = _MealOptions(portions("dairy", (100, 175, 250)), fruit_portions, fat_portions)
    return breakfast, main, main, snack


class StarterPlanRecommender:

    def __init__(self, foods):
        self._foods = reviewed_foods(foods)
        self._indexes = {}

    def generate(self, profile, *, meal_shares=MEAL_SHARES, excluded_ids=()):
        if (len(meal_shares) != len(MEALS) or not all(share > 0 for share in meal_shares)
                or not isclose(sum(meal_shares), 1.0)):
            raise ValueError("Meal shares must be positive and add up to 100%.")
        targets = profile.targets()
        target = np.array([targets[key] for key in NUTRIENTS[:4]])
        settings = profile.data
        diet = settings["diet"]
        allowed = eligible_foods(self._foods, profile)
        if excluded_ids:
            allowed = allowed.loc[~allowed["food_id"].isin(excluded_ids)]
            indexes = _fit_meals(allowed)
        else:
            if diet not in self._indexes:
                self._indexes[diet] = _fit_meals(allowed)
            indexes = self._indexes[diet]

        plan, used_foods, planned_calories = MealPlan(), set(), 0.0
        for meal, share, options in zip(MEALS, meal_shares, indexes):
            minimum = max(0, MIN_ESTIMATE_CALORIES - planned_calories) if meal == "Snack" else 0
            index = options.closest(target * share, used_foods, settings["preference"], minimum)
            for food_id, grams in zip(options.ids[index], options.grams[index]):
                plan.add(food_id, float(grams), meal, allowed)
                used_foods.add(food_id)
            planned_calories += options.nutrients[index, 0]
        return plan
