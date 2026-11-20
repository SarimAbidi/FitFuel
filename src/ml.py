from collections import defaultdict
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from .config import RESULTS
from .data import write_json


def clean_text(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


def food_name_key(text):
    """Match complete names while preserving details such as fat percentages."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized))


def food_category_lookup(text_data, foods):
    categories = defaultdict(set)
    for data, name_column in ((text_data, "description"), (foods, "name")):
        for name, category in data[[name_column, "category"]].itertuples(index=False, name=None):
            categories[food_name_key(name)].add(category)

    aliases = {
        "01129": ("egg", "eggs", "boiled egg", "boiled eggs", "hard-boiled eggs"),
        "01256": ("plain Greek yogurt",),
        "05064": ("chicken breast", "roasted chicken breast"),
        "09040": ("fresh banana", "raw banana"),
    }
    catalogue_categories = foods.set_index("food_id")["category"].to_dict()
    for food_id, names in aliases.items():
        if food_id in catalogue_categories:
            for name in names:
                categories[food_name_key(name)].add(catalogue_categories[food_id])
    return {key: next(iter(labels)) for key, labels in categories.items() if key and len(labels) == 1}


def food_family(description):
    """Keep related preparations together, using a simple USDA-name heuristic."""
    parts = description.split(",")
    first = clean_text(parts[0])
    if first in ("nuts", "seeds", "beans"):
        return clean_text(",".join(parts[:2]))
    if first.startswith("cereals"):
        return clean_text(",".join(parts[:3]))
    return first


def train_models(text_data, results_dir=RESULTS):
    """Fit both pipelines only on training families and evaluate one held-out split."""
    data = text_data.copy()
    data["clean_text"] = data["description"].map(clean_text)
    data = data[data["clean_text"].ne("")].drop_duplicates("clean_text").reset_index(drop=True)
    data["family"] = data["description"].map(food_family)
    labels = sorted(set(data["category"]))
    splitter = GroupShuffleSplit(n_splits=30, test_size=0.20, random_state=42)
    split = None
    for attempt, (train, test) in enumerate(splitter.split(data, groups=data["family"]), 1):
        if set(data.iloc[train]["category"]) == set(labels) == set(data.iloc[test]["category"]):
            split = (train, test)
            break
    if split is None:
        raise ValueError("Not enough food families to put every class in train and test.")
    train, test = split
    x_train, x_test = data.iloc[train]["description"], data.iloc[test]["description"]
    y_train, y_test = data.iloc[train]["category"], data.iloc[test]["category"]
    classifiers = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                   "Multinomial Naive Bayes": MultinomialNB(alpha=1.0)}
    models, metrics, predictions = {}, [], {}
    reports = {}
    for name, classifier in classifiers.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", classifier)])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        models[name] = pipeline
        predictions[name] = predicted.tolist()
        metrics.append({"model": name, "accuracy": accuracy_score(y_test, predicted),
                        "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0)})
        reports[name] = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    default = "Logistic Regression"
    matrix = confusion_matrix(y_test, predictions[default], labels=labels)
    majority = y_train.mode().iloc[0]
    result = {"metrics": metrics, "default_model": default, "labels": labels,
              "confusion_matrix": matrix.tolist(), "classification_reports": reports,
              "train_rows": len(train), "test_rows": len(test),
              "train_families": data.iloc[train]["family"].nunique(),
              "test_families": data.iloc[test]["family"].nunique(),
              "split_attempt": attempt, "random_state": 42,
              "split": "20% of food families held out; row fraction can differ",
              "majority_baseline_accuracy": float((y_test == majority).mean()),
              "train_class_counts": y_train.value_counts().to_dict(),
              "test_class_counts": y_test.value_counts().to_dict(),
              "limitations": "Heuristic families, one split, English USDA descriptions; not a health classifier."}
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "model_results.json", result)
    pd.DataFrame(metrics).to_csv(results_dir / "model_comparison.csv", index=False)
    assignments = data[["food_id", "description", "category", "family"]].copy()
    assignments["split"] = "train"
    assignments.loc[test, "split"] = "test"
    assignments.to_csv(results_dir / "split_assignments.csv", index=False)
    held_out = data.iloc[test][["food_id", "description", "category"]].copy()
    for name, predicted in predictions.items():
        held_out[name] = predicted
    held_out.to_csv(results_dir / "test_predictions.csv", index=False)
    return models[default], result


def predict_category(model, description):
    """Abstain on empty/unknown vocabulary or low-confidence input."""
    cleaned = clean_text(description)
    if not cleaned:
        raise ValueError("Enter a food description containing letters.")
    vector = model.named_steps["tfidf"].transform([description])
    if vector.nnz == 0:
        return "Unknown (no familiar food words)", 0.0
    probabilities = model.predict_proba([description])[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    if confidence < 0.40:
        return "Uncertain - try a more specific food description", confidence
    return str(model.classes_[index]), confidence
