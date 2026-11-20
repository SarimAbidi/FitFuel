from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import requests

from .config import CACHE, RAW
from .data import write_json


class FoodAPIError(ValueError):
    """A lookup has neither a valid live reply nor a usable cached reply."""


def validate_reply(reply, barcode):
    if not isinstance(reply, dict) or not isinstance(reply.get("product"), dict):
        raise FoodAPIError("API did not return a product.")
    product = reply["product"]
    if str(product.get("code", reply.get("code", ""))) != barcode:
        raise FoodAPIError("API returned a different barcode.")
    if not isinstance(product.get("product_name"), str) or not product["product_name"].strip():
        raise FoodAPIError("This product has no usable name.")
    return product


def read_cached(path, barcode):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or not isinstance(envelope.get("fetched_at"), str):
        raise FoodAPIError("Malformed cache metadata.")
    validate_reply(envelope.get("reply"), barcode)
    return envelope


def lookup_food(barcode, offline=False, cache_dir=CACHE, sample_dir=RAW):
    """Try live data, then the local cache, then the shipped real API sample."""
    if not isinstance(barcode, str) or not re.fullmatch(r"\d{8,14}", barcode):
        raise FoodAPIError("Enter an 8-14 digit barcode.")
    filename = f"off_{barcode}.json"
    error = "Offline mode selected."
    if not offline:
        url = f"https://world.openfoodfacts.org/api/v3/product/{barcode}"
        try:
            response = requests.get(
                url, params={"fields": "code,product_name,nutriments,nutrition,nutrition_data_per,quantity"},
                headers={"User-Agent": os.getenv("FITFUEL_USER_AGENT", "FitFuel-Student-Project/1.0 (educational console application)")},
                timeout=10)
            if response.status_code != 200:
                response.raise_for_status()
                raise FoodAPIError(f"Unexpected HTTP status {response.status_code}.")
            reply = response.json()
            validate_reply(reply, barcode)
            envelope = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                        "source_url": response.url, "reply": reply}
            warning = ""
            try:
                write_json(Path(cache_dir) / filename, envelope)
            except OSError as exc:
                warning = f"Live result available, but cache could not be saved: {exc}"
            return envelope, "live", warning
        except (requests.RequestException, ValueError) as exc:
            error = f"Live lookup unavailable: {exc}"
    for directory, source in ((cache_dir, "local cache"), (sample_dir, "bundled API snapshot")):
        try:
            return read_cached(Path(directory) / filename, barcode), source, error
        except (OSError, ValueError, TypeError):
            continue
    raise FoodAPIError(f"{error} No cached product found. Local meal planning still works.")


def nutrition_values(product):
    """Return labelled API values; missing information remains None, not zero."""
    fields = {"calories": "energy-kcal", "protein": "proteins", "carbs": "carbohydrates",
              "fat": "fat", "fibre": "fiber", "sugar": "sugars"}
    legacy = product.get("nutriments") or {}
    nutrition = product.get("nutrition") or {}
    aggregate = nutrition.get("aggregated_set", {}) if isinstance(nutrition, dict) else {}
    if not isinstance(aggregate, dict):
        aggregate = {}
    structured = aggregate.get("nutrients", {})
    if not isinstance(structured, dict):
        structured = {}
    result = {}
    for name, key in fields.items():
        value = legacy.get(key + "_100g") if isinstance(legacy, dict) else None
        if value is None and aggregate.get("per") == "100g":
            item = structured.get(key, {})
            if isinstance(item, dict) and item.get("unit") == ("kcal" if name == "calories" else "g"):
                value = item.get("value")
        try:
            value = float(value)
            result[name] = value if math.isfinite(value) and value >= 0 else None
        except (TypeError, ValueError):
            result[name] = None
    return result
