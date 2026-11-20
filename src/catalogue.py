import pandas as pd

from .config import STARTER_ROLES


def build_catalogue(nutrition, annotations_path):
    """Keep every complete USDA food and overlay reviewed metadata by ID."""
    annotations = pd.read_csv(annotations_path, dtype={"food_id": str})
    if not {"food_id", "name", "vegetarian"}.issubset(annotations.columns):
        raise ValueError("Planner catalogue is missing required columns.")
    if annotations["food_id"].duplicated().any():
        raise ValueError("Planner food IDs must be unique.")
    annotations["name"] = annotations["name"].astype("string").str.strip()
    if annotations["name"].isna().any() or annotations["name"].eq("").any():
        raise ValueError("Planner food names must be nonempty.")
    flags = annotations["vegetarian"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Vegetarian values must be true or false.")
    annotations["vegetarian"] = flags.eq("true")
    if "starter_role" in annotations and not annotations["starter_role"].isin(STARTER_ROLES).all():
        raise ValueError("Planner catalogue contains an unknown starter plan role.")

    reviewed = annotations.merge(nutrition, on="food_id", how="inner", validate="one_to_one")
    unreviewed = nutrition.loc[~nutrition["food_id"].isin(reviewed["food_id"])].copy()
    unreviewed["name"] = unreviewed["description"]
    unreviewed["vegetarian"] = False
    if "starter_role" in reviewed:
        unreviewed["starter_role"] = "manual"
    foods = pd.concat([reviewed.assign(reviewed=True), unreviewed.assign(reviewed=False)], ignore_index=True)
    if foods.empty:
        raise ValueError("No usable foods remain after cleaning.")
    return foods, len(annotations), len(reviewed)


def reviewed_foods(foods):
    return foods.loc[foods["reviewed"]] if "reviewed" in foods else foods


def eligible_foods(foods, profile):
    return foods.loc[foods["vegetarian"]] if profile.data["diet"] == "vegetarian" else foods


def search_foods(foods, query):
    """Find all query words in the food name, source description, category or ID."""
    words = query.casefold().split()
    if not words:
        return foods
    searchable = foods["name"].str.casefold()
    for column in ("description", "category", "food_id"):
        if column in foods:
            searchable = searchable.str.cat(foods[column].str.casefold(), sep=" ", na_rep="")
    matches = pd.Series(True, index=foods.index)
    for word in words:
        matches &= searchable.str.contains(word, regex=False, na=False)
    return foods.loc[matches]
