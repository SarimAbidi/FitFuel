import argparse
from dataclasses import replace
import json
import math
import sqlite3
import sys
import pandas as pd

from .catalogue import eligible_foods
from .config import (ACTIVITY, GOALS, MIN_LOSS_BMI, NUTRIENTS, PREFERENCES,
                     PRIVATE, RESULTS, WEIGHT_RANGE, make_directories)
from .console import ask_number, choose
from .data import load_data, load_plan, save_plan, write_json
from .food_browser import browse_database, show_foods
from .meal_menu import customize_starter_plan, edit_plan, show_plan
from .models import UserProfile, InvalidProfileError, MealPlan, GoalBasedRecommender, Recommender
from .ml import food_category_lookup, food_name_key, train_models, predict_category
from .api import lookup_food, nutrition_values
from .charts import save_charts
from .profiles import ProfileStore, SavedProfile
from .starter_plan import StarterPlanRecommender

NOTICE = ("General wellness estimates for adults aged 19-78. Not medical nutrition advice.\n"
          "Not intended for pregnancy, breastfeeding or medically prescribed diets.")


def create_profile():
    print(NOTICE)
    age = ask_number("Age (years): ", 19, 78, whole=True)
    weight = ask_number("Current weight (kg): ", *WEIGHT_RANGE)
    height = ask_number("Height (cm): ", 130, 220)
    sex = choose("Sex coefficient used by the published formula:", ("male", "female"))
    activity = choose("Activity (sedentary = mostly sitting; active = frequent exercise):", ACTIVITY)
    goal = choose("General goal:", GOALS)
    target_weight = None
    if goal != "maintain":
        lowest = max(WEIGHT_RANGE[0], MIN_LOSS_BMI * (height / 100) ** 2)
        highest = WEIGHT_RANGE[1]
        limit = weight - lowest if goal == "lose" else highest - weight
        limit = math.floor(limit * 10) / 10
        if limit < 0.1:
            raise InvalidProfileError(f"A goal to {goal} weight is outside the supported range for this profile.")
        change = ask_number(f"How many kg do you want to {goal}? (0.1 to {limit:g}): ", 0.1, limit)
        target_weight = max(lowest, weight - change) if goal == "lose" else min(highest, weight + change)
    diet = choose("Food preference (vegetarian includes dairy, excludes eggs):", ("vegetarian", "non-vegetarian"))
    preference = choose("Food ranking preference:", PREFERENCES, default="balanced")
    profile = UserProfile(age, weight, height, sex, activity, goal, diet, preference,
                          target_weight=target_weight)
    profile.targets()
    return profile


def display_targets(profile):
    targets = profile.targets()
    print("\nDAILY ESTIMATES")
    settings = profile.data
    if "target_weight" in settings:
        weight, target_weight = settings["weight"], settings["target_weight"]
        print(f"Weight goal: {settings['goal']} {abs(target_weight - weight):g} kg "
              f"({weight:g} to {target_weight:g} kg)")
    print(f"Resting energy: {targets['resting']:.0f} kcal")
    print(f"Maintenance:   {targets['maintenance']:.0f} kcal")
    for key in ("calories", "protein", "carbs", "fat"):
        unit = "kcal" if key == "calories" else "g"
        print(f"{key.title():12} {targets[key]:8.1f} {unit}")
    print("Macros use 25% protein, 45% carbohydrate and 30% fat: a simple project assumption.")


def show_api(barcode, offline=False):
    envelope, source, warning = lookup_food(barcode, offline=offline)
    product = envelope["reply"]["product"]
    print(f"\n{product['product_name']} | Barcode: {barcode}")
    print(f"Source: {source} | Fetched: {envelope['fetched_at']}")
    if warning:
        print(warning)
    print("Open Food Facts: per 100 g / 100 ml as supplied by the product label.")
    for key, value in nutrition_values(product).items():
        unit = "kcal" if key == "calories" else "g"
        shown = "unknown" if value is None else f"{value:g} {unit}"
        print(f"{key.title():12}: {shown}")
    print("Community-supplied label data; lookup only. Check the package for ingredients/allergens.")


def show_model_results(result):
    print(f"\nTraining rows: {result['train_rows']} | Test rows: {result['test_rows']}")
    print("Split: separate food families; both models use exactly the same test rows.")
    for row in result["metrics"]:
        print(f"{row['model']:25} Accuracy: {row['accuracy']:.3f}  Macro-F1: {row['macro_f1']:.3f}")
    print(f"Majority-class baseline accuracy: {result['majority_baseline_accuracy']:.3f}")
    print("Default: Logistic Regression, chosen before evaluation. No health suitability is predicted.")


def run_demo(text, foods, summary, offline=False):
    print("\nDEMO: fictional adult profile; illustrative portions, not a recommended full-day diet.")
    profile = UserProfile(25, 78, 175, "male", "moderate", "lose", target_weight=73)
    plan = StarterPlanRecommender(foods).generate(profile)
    display_targets(profile)
    strategy: Recommender = GoalBasedRecommender()
    show_foods(strategy.recommend(foods, profile))
    show_plan(plan, profile, foods, heading="SUGGESTED STARTER PLAN")
    save_plan(RESULTS / "demo_plan.json", profile, plan)
    restored_profile, restored_plan = load_plan(RESULTS / "demo_plan.json", foods)
    print(f"Save/load check: restored {len(restored_plan.items)} entries.")
    model, result = train_models(text)
    show_model_results(result)
    for example in ("fresh banana", "roasted chicken breast", "plain yogurt", "xyzqwerty"):
        category, confidence = predict_category(model, example)
        print(f"{example!r} -> {category} (model score {confidence:.2f})")
    for path in save_charts(restored_profile, restored_plan, foods, summary, result):
        print(f"Chart saved: {path.name}")
    show_api("3017620422003", offline=offline)
    write_json(RESULTS / "demo_totals.json", dict(zip(NUTRIENTS, plan.totals(foods).tolist())))


def new_person(store, imported=None):
    name = input("Profile name (Enter to cancel): ").strip()
    if not name:
        return None
    if any(name.casefold() == saved.casefold() for saved in store.names()):
        raise ValueError("A profile with that name already exists. Open it with option 8.")
    profile, plan = imported if imported is not None else (create_profile(), MealPlan())
    return store.save(SavedProfile(name, profile, plan).record_weight(profile.data["weight"]))


def open_person(store, legacy_path, foods):
    choices = {f"Open {name}": name for name in store.names()}
    actions = list(choices)
    if legacy_path.is_file():
        actions.append("Import old saved meal plan")
    if not actions:
        print("No saved profiles yet. Create one with option 1.")
        return None
    choice = choose("Saved profiles", [*actions, "Back"])
    if choice in choices:
        return store.load(choices[choice], foods)
    if choice == "Import old saved meal plan":
        return new_person(store, imported=load_plan(legacy_path, foods))
    return None


def update_person(store, person, foods):
    profile = create_profile()
    allowed = eligible_foods(foods, profile)
    plan = MealPlan()
    allowed_ids = set(allowed["food_id"])
    for item in person.plan.items:
        if item["food_id"] in allowed_ids:
            plan.add(foods=allowed, **item)
    removed = len(person.plan.items) - len(plan.items)
    if removed and input(f"Apply this diet and remove {removed} incompatible entries? (y/n): ").strip().lower() != "y":
        return None
    updated = replace(person, profile=profile, plan=plan)
    weight = profile.data["weight"]
    if weight != person.profile.data["weight"] and (not person.check_ins or weight != person.check_ins[-1][1]):
        updated = updated.record_weight(weight)
    return store.save(updated)


def show_weight_history(person):
    print(f"\nWeight history for {person.name}")
    for day, weight in person.check_ins:
        print(f"{day}  {weight:g} kg")
    if len(person.check_ins) > 1:
        change = person.check_ins[-1][1] - person.check_ins[0][1]
        print(f"Change since first check-in: {change:+g} kg")
    target = person.profile.data.get("target_weight")
    if target is not None:
        print(f"Target weight: {target:g} kg")


def run_menu(text, foods, summary, offline=False):
    person, model, result = None, None, None
    store = ProfileStore(PRIVATE / "profiles.sqlite3")
    legacy_path = PRIVATE / "meal_plan.json"
    strategy: Recommender = GoalBasedRecommender()
    starter = StarterPlanRecommender(foods)
    known_categories = food_category_lookup(text, foods)

    def save_edit(plan):
        nonlocal person
        person = store.save(replace(person, plan=plan))

    if store.names() or legacy_path.is_file():
        try:
            person = open_person(store, legacy_path, foods)
        except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
            print(f"Could not open a saved profile: {exc}")
    while True:
        profile = person.profile if person else None
        print(f"\nFitFuel{' | ' + person.name if person else ''}\n"
              "1. Set up or update profile\n2. View nutrition targets\n3. Get a starter plan or food suggestions\n"
              "4. Edit or view meal plan\n5. Look up a barcode\n6. Classify a food description\n"
              "7. View model results and charts\n8. Profiles and weight history\n"
              "9. Browse the food database\n0. Exit")
        choice = input("Choice: ").strip()
        try:
            if choice == "0":
                print("Goodbye.")
                return
            if choice == "1":
                updated = update_person(store, person, foods) if person else new_person(store)
                if updated:
                    person = updated
                    print(f"Saved profile for {person.name}.")
                    display_targets(person.profile)
            elif choice in ("2", "3", "4"):
                if profile is None:
                    print("Create a profile with option 1, or open a saved profile with option 8.")
                    continue
                if choice == "2":
                    display_targets(profile)
                elif choice == "3":
                    action = choose("Suggestions", ("Starter plan", "Browse foods", "Back", "Customize starter plan"),
                                    default="Starter plan")
                    if action in ("Starter plan", "Customize starter plan"):
                        draft = (customize_starter_plan(starter, profile, foods) if action == "Customize starter plan"
                                 else starter.generate(profile))
                        show_plan(draft, profile, foods, heading="STARTER PLAN PREVIEW")
                        prompt = "Replace the current plan with this starter plan?" if person.plan.items else "Use this starter plan?"
                        if input(f"{prompt} (y/n): ").strip().lower() == "y":
                            save_edit(draft)
                            print("Starter plan added and saved. Use option 4 to edit it.")
                    elif action == "Browse foods":
                        categories = ["All categories"] + sorted(set(foods["category"]))
                        category = choose("Filter suggestions:", categories, default="All categories")
                        chosen = None if category == "All categories" else category
                        show_foods(strategy.recommend(foods, profile, category=chosen))
                else:
                    edit_plan(person.plan, profile, foods, save_edit)
            elif choice == "5":
                barcode = input("Barcode (Enter = bundled example 3017620422003): ").strip() or "3017620422003"
                show_api(barcode, offline)
            elif choice in ("6", "7"):
                if choice == "6":
                    description = input("Food name and preparation (e.g. hard-boiled egg): ").strip()
                    category = known_categories.get(food_name_key(description))
                    if category is not None:
                        print(f"Food category: {category} (catalogue match)")
                        continue
                if model is None:
                    print("Training two small text models on the local data...")
                    model, result = train_models(text)
                if choice == "6":
                    category, score = predict_category(model, description)
                    if category in model.classes_:
                        print(f"Predicted category: {category} | Model score: {score:.2f}")
                    else:
                        status = "Uncertain" if score else "Unknown food"
                        print(f"{status}. Enter one food and how it was prepared, such as roasted chicken breast.")
                else:
                    show_model_results(result)
                    if profile is None:
                        print("Create or load a profile to generate the three charts.")
                    else:
                        for path in save_charts(profile, person.plan, foods, summary, result):
                            print(f"Chart saved: {path}")
            elif choice == "8":
                action = choose("Profiles and weight history", ("Open or import a profile", "Create another profile",
                                                                  "Record weight", "View weight history", "Back"))
                if action == "Open or import a profile":
                    person = open_person(store, legacy_path, foods) or person
                elif action == "Create another profile":
                    person = new_person(store) or person
                elif action in ("Record weight", "View weight history"):
                    if person is None:
                        print("Create or open a profile first.")
                    elif action == "Record weight":
                        weight = ask_number("Weight for this check-in (kg): ", *WEIGHT_RANGE)
                        person = store.save(person.record_weight(weight))
                        print("Check-in saved. To update your nutrition estimates, update your profile with option 1.")
                    else:
                        show_weight_history(person)
            elif choice == "9":
                browse_database(foods)
            else:
                print("Choose a number from the menu.")
        except FileNotFoundError:
            print("File not found. Save a plan first, or restore the bundled data folder.")
        except json.JSONDecodeError:
            print("Invalid JSON file. Your current session has not changed.")
        except (ValueError, TypeError, KeyError) as exc:
            print(f"Could not complete that action: {exc}")
        except (OSError, sqlite3.Error) as exc:
            print(f"Could not read/write the local file: {exc}")


def main():
    parser = argparse.ArgumentParser(description="FitFuel - simple console nutrition planner")
    parser.add_argument("--offline", action="store_true", help="Use cached API data without a network request")
    parser.add_argument("--demo", action="store_true", help="Run a complete fictional example without input")
    parser.add_argument("--train-only", action="store_true", help="Train both models and export evaluation files")
    args = parser.parse_args()
    try:
        make_directories()
        text, foods, summary, stats = load_data()
        print(NOTICE)
        print(f"Loaded {len(text):,} labelled records and {len(foods):,} searchable foods.")
        if stats["catalogue_usable"] != stats["catalogue_requested"]:
            print("Some planner entries were excluded due to missing/invalid nutrient data.")
        if args.train_only:
            _, result = train_models(text)
            show_model_results(result)
        elif args.demo:
            run_demo(text, foods, summary, args.offline)
        else:
            run_menu(text, foods, summary, args.offline)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nSession ended.")
        return 0
    except FileNotFoundError as exc:
        print(f"Required data file missing: {exc.filename}. Restore data/raw from the ZIP.")
        return 1
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error, pd.errors.ParserError) as exc:
        print(f"Unable to start/complete FitFuel: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
