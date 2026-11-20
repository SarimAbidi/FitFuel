from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRIVATE = ROOT / "data" / "private"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"

NUTRIENTS = ("calories", "protein", "carbs", "fat", "fibre", "sugar")
MEALS = ("Breakfast", "Lunch", "Dinner", "Snack")
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
GOALS = {"lose": 0.90, "maintain": 1.00, "gain": 1.10}
WEIGHT_RANGE = (40, 250)
MIN_LOSS_BMI = 18.5
MIN_ESTIMATE_CALORIES = 1200
MAX_PLAN_ITEMS = 200
PREFERENCES = ("balanced", "high-protein", "high-fibre", "lower-sugar")
STARTER_ROLES = ("breakfast_grain", "dairy", "fruit", "fat", "protein", "starch", "vegetable", "manual")


def make_directories():
    for folder in (PROCESSED, PRIVATE, CACHE, RESULTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
