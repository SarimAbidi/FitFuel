import json
from pathlib import Path
import numpy as np
import pandas as pd

from .catalogue import build_catalogue, eligible_foods
from .config import MAX_PLAN_ITEMS, NUTRIENTS, RAW, PROCESSED
from .models import MealPlan, UserProfile


def write_json(path, value):
    """Write JSON through a temporary file to avoid half-written saves."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_data(raw_dir=RAW, output_dir=PROCESSED):
    """Keep labelled text for ML; drop unknown nutrients from numerical analysis."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    raw = pd.read_csv(raw_dir / "usda_foods.csv", dtype={"food_id": str})
    required = {"food_id", "description", "category", *NUTRIENTS}
    if not required.issubset(raw.columns):
        raise ValueError("Nutrition CSV is missing required columns.")
    text = raw.dropna(subset=["food_id", "description", "category"]).copy()
    text["description"] = text["description"].astype(str).str.strip()
    text = text[text["description"].ne("")].drop_duplicates("food_id")
    numeric = text.copy()
    numeric[list(NUTRIENTS)] = numeric[list(NUTRIENTS)].apply(pd.to_numeric, errors="coerce")
    numeric[list(NUTRIENTS)] = numeric[list(NUTRIENTS)].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(subset=list(NUTRIENTS))
    numeric = numeric[(numeric[list(NUTRIENTS)] >= 0).all(axis=1)]
    foods, annotated_count, reviewed_count = build_catalogue(numeric, raw_dir / "planner_catalogue.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric.to_csv(output_dir / "cleaned_foods.csv", index=False)
    summary = numeric.groupby("category").agg(
        foods=("food_id", "count"), mean_calories=("calories", "mean"),
        mean_protein=("protein", "mean"), mean_fibre=("fibre", "mean"))
    summary.to_csv(output_dir / "category_summary.csv")
    stats = {"raw_rows": len(raw), "labelled_rows": len(text),
             "complete_nutrient_rows": len(numeric),
             "excluded_nutrient_rows": len(text) - len(numeric),
             "catalogue_requested": annotated_count, "catalogue_usable": reviewed_count,
             "database_foods": len(foods),
             "missing_policy": "Drop unknown/invalid nutrients for calculations; never fill with zero."}
    write_json(output_dir / "cleaning_summary.json", stats)
    return text, foods, summary, stats


def save_plan(path, profile, plan):
    """Save only the profile, IDs and portions, not untrusted nutrient totals."""
    write_json(path, plan_payload(profile, plan))


def plan_payload(profile, plan):
    return {"version": 1, "profile": profile.data, "items": plan.items}


def load_plan(path, foods):
    return parse_plan(json.loads(Path(path).read_text(encoding="utf-8")), foods)


def parse_plan(payload, foods):
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported plan format.")
    if not isinstance(payload.get("profile"), dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Plan needs a profile and an item list.")
    if len(payload["items"]) > MAX_PLAN_ITEMS:
        raise ValueError("Plan contains too many entries.")
    profile = UserProfile(**payload["profile"])
    profile.targets()
    plan = MealPlan()
    allowed = eligible_foods(foods, profile)
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"food_id", "grams", "meal"}:
            raise ValueError("Malformed meal entry.")
        plan.add(item["food_id"], item["grams"], item["meal"], allowed)
    return profile, plan
