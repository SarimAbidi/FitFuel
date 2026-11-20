from textwrap import wrap

from .catalogue import search_foods


def show_foods(foods):
    if foods.empty:
        print("No matching foods.")
        return
    print("\nValues PER 100 g. Enter the weight of the food in its named cooked/raw state.")
    print(f"{'ID':5}  {'Food':47} {'kcal':>7} {'P(g)':>6} {'C(g)':>6} {'F(g)':>6}")
    for food in foods.to_dict("records"):
        name, *continuation = wrap(food["name"], width=47)
        print(f"{food['food_id']:5}  {name:47} {food['calories']:7.1f} "
              f"{food['protein']:6.1f} {food['carbs']:6.1f} {food['fat']:6.1f}")
        for line in continuation:
            print(f"       {line}")


def select_food(matches, selection_prompt="Food ID"):
    if matches.empty:
        show_foods(matches)
        return None
    matches = matches.sort_values("name", key=lambda names: names.str.casefold())
    page_size = 20
    pages = (len(matches) + page_size - 1) // page_size
    page = 0
    while True:
        start = page * page_size
        visible = matches.iloc[start:start + page_size]
        show_foods(visible)
        if pages > 1:
            print(f"Page {page + 1} of {pages} ({len(matches):,} foods)")
            prompt = f"{selection_prompt}, N for next, P for previous, or Enter to go back: "
        else:
            prompt = f"{selection_prompt} (Enter to go back): "
        answer = input(prompt).strip()
        if not answer:
            return None
        if answer.casefold() == "n" and page + 1 < pages:
            page += 1
        elif answer.casefold() == "p" and page > 0:
            page -= 1
        elif answer in set(visible["food_id"]):
            return answer
        else:
            print("Choose a displayed food ID or a page with more results.")


def browse_database(foods):
    print(f"\nFood database | {len(foods):,} foods")
    query = input("Search by name, category or ID (Enter = all foods): ").strip()
    matches = search_foods(foods, query)
    food_id = select_food(matches, selection_prompt="Food ID for details")
    if food_id is None:
        return
    selected = matches.loc[matches["food_id"].eq(food_id)]
    show_foods(selected)
    food = selected.iloc[0]
    if not food["reviewed"]:
        diet_label = "Vegetarian status unreviewed"
    elif food["vegetarian"]:
        diet_label = "Available for vegetarian profiles"
    else:
        diet_label = "Available for non-vegetarian profiles"
    print(f"{food['category']} | {diet_label}")
    print(f"Fibre {food['fibre']:g} g | Sugar {food['sugar']:g} g per 100 g")
