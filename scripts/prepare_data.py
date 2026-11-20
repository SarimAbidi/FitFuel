"""Rebuild the compact USDA CSV from the official SR Legacy ASCII ZIP.

The download is optional: the ready-to-run dataset is already bundled.
Usage: python scripts/prepare_data.py --archive path/to/SR-Leg_ASC.zip
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.ars.usda.gov/ARSUserFiles/80400525/Data/SR-Legacy/SR-Leg_ASC.zip"
GROUPS = {"0100": "Dairy and eggs", "0500": "Poultry", "0800": "Grains and cereals",
          "0900": "Fruit", "1100": "Vegetables", "1200": "Nuts and seeds",
          "1600": "Legumes", "2000": "Grains and cereals"}
NUTRIENTS = {"208": "calories", "203": "protein", "205": "carbs",
             "204": "fat", "291": "fibre", "269": "sugar"}


def rows(archive, filename):
    text = archive.read(filename).decode("latin-1")
    return csv.reader(io.StringIO(text), delimiter="^", quotechar="~")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.archive:
        content = args.archive.read_bytes()
    else:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        content = response.content
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        group_names = {r[0]: r[1] for r in rows(archive, "FD_GROUP.txt")}
        foods = {}
        for r in rows(archive, "FOOD_DES.txt"):
            if r[1] in GROUPS:
                foods[r[0]] = {"food_id": r[0], "description": r[2],
                              "category": GROUPS[r[1]], "usda_group_code": r[1],
                              "usda_group": group_names[r[1]],
                              **{name: "" for name in NUTRIENTS.values()}}
        for r in rows(archive, "NUT_DATA.txt"):
            if r[0] in foods and r[1] in NUTRIENTS:
                foods[r[0]][NUTRIENTS[r[1]]] = r[2]
    destination = ROOT / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "usda_foods.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(next(iter(foods.values()))))
        writer.writeheader()
        writer.writerows(foods.values())
    metadata = {"source": "USDA National Nutrient Database for Standard Reference, Legacy Release (2018)",
                "url": URL, "doi": "https://doi.org/10.15482/USDA.ADC/1529216",
                "source_archive_sha256": hashlib.sha256(content).hexdigest(),
                "food_rows": len(foods), "units": "kcal and grams per 100 g edible portion",
                "group_mapping": GROUPS, "nutrient_mapping": NUTRIENTS,
                "missing_values": "Preserved as empty fields; no invented nutrient values."}
    (destination / "source_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(foods)} source rows to {destination}.")


if __name__ == "__main__":
    main()
