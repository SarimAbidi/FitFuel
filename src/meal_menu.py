from .catalogue import eligible_foods, reviewed_foods, search_foods
from .config import MEALS, NUTRIENTS
from .console import ask_number, choose
from .food_browser import select_food
from .starter_plan import MEAL_SHARES


def customize_starter_plan(starter, profile, foods):
    available = eligible_foods(reviewed_foods(foods), profile)
    if "starter_role" in available:
        available = available.loc[available["starter_role"].ne("manual")]
    excluded = set()
    while True:
        query = input("Search for a food to leave out (Enter when done): ").strip()
        if not query:
            break
        remaining = available.loc[~available["food_id"].isin(excluded)]
        matches = search_foods(remaining, query)
        food_id = select_food(matches, selection_prompt="Food ID to leave out")
        if food_id is not None:
            excluded.add(food_id)

    while True:
        entered = input("Calorie target percentages for breakfast, lunch, dinner and snack "
                        "(Enter for 25 30 30 15): ").strip()
        if not entered:
            shares = MEAL_SHARES
            break
        try:
            percentages = [int(value) for value in entered.replace(",", " ").split()]
        except ValueError:
            percentages = []
        if len(percentages) == len(MEALS) and all(value > 0 for value in percentages) and sum(percentages) == 100:
            shares = tuple(value / 100 for value in percentages)
            break
        print("Enter four positive whole percentages that add up to 100.")
    return starter.generate(profile, meal_shares=shares, excluded_ids=excluded)


def show_plan(plan, profile, foods, heading="CURRENT DAILY PLAN"):
    print(f"\n{heading}")
    names = foods.set_index("food_id")["name"].to_dict()
    for index, item in enumerate(plan.items, 1):
        print(f"{index}. {item['meal']:10} {names[item['food_id']]} - {item['grams']:g} g")
    if not plan.items:
        print("Your plan is empty. Add foods to begin.")
    else:
        for meal in MEALS:
            total = plan.totals(foods, meal=meal)
            print(f"{meal:10}: {total[0]:.0f} kcal, {total[1]:.1f} g protein")
    totals = plan.totals(foods)
    targets = profile.targets()
    print("\nNutrient       Selected     Estimated target")
    for key, value in zip(NUTRIENTS, totals):
        unit = "kcal" if key == "calories" else "g"
        target = f"{targets[key]:.1f} {unit}" if key in targets else "not set"
        print(f"{key.title():12} {value:8.1f} {unit:4}  {target}")
    print("A partial plan is not a complete daily diet. Fibre and sugar are totals only.")


def edit_plan(plan, profile, foods, save_edit):
    allowed = eligible_foods(foods, profile)
    while True:
        action = choose("\nMeal plan:", ("View plan", "Add food", "Remove food", "Change portion", "Back"))
        if action == "Back":
            return
        if action == "View plan":
            show_plan(plan, profile, foods)
        elif action == "Add food":
            if len(allowed) < len(foods):
                print("Showing reviewed vegetarian foods. Main menu option 9 searches the full database.")
            query = input(f"Search {len(allowed):,} foods by name, category or ID (Enter = all): ").strip()
            matches = search_foods(allowed, query)
            food_id = select_food(matches)
            if not food_id:
                continue
            grams = ask_number("Portion weight (g): ", 1, 1500)
            meal = choose("Meal:", MEALS)
            draft = plan.copy()
            draft.add(food_id, grams, meal, allowed)
            save_edit(draft)
            plan = draft
            print("Food added and saved.")
        elif action in ("Remove food", "Change portion"):
            if not plan.items:
                print("Your plan is already empty.")
                continue
            show_plan(plan, profile, foods)
            number = ask_number("Entry number: ", 1, len(plan.items), whole=True)
            draft = plan.copy()
            if action == "Remove food":
                draft.remove(number)
                message = "Entry removed and saved."
            else:
                grams = ask_number("New portion weight (g): ", 1, 1500)
                draft.change_portion(number, grams)
                message = "Portion updated and saved."
            save_edit(draft)
            plan = draft
            print(message)
