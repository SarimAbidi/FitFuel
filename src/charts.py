from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES


def save_charts(profile, plan, foods, summary, model_result, directory=FIGURES):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    targets, total = profile.targets(), plan.totals(foods)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(bottom=0.21, top=0.88, left=0.12, right=0.97)
    bars = ax.bar(["Estimated target", "Selected foods"], [targets["calories"], total[0]],
                  color=["#087f8c", "#f0a34b"])
    ax.bar_label(bars, fmt="%.0f", padding=4)
    ax.set(title="Daily calorie estimate and current plan", ylabel="Energy (kcal)", xlabel="Daily total")
    ax.set_ylim(0, max(targets["calories"], total[0], 1) * 1.18)
    fig.text(0.5, 0.01, "Selected foods may be an incomplete day; estimates are not medical advice.", ha="center", fontsize=9)
    path = directory / "01_calorie_comparison.png"
    fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ordered = summary.sort_values("mean_protein")
    bars = ax.barh(ordered.index, ordered["mean_protein"], color="#087f8c")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set(title="Mean protein by source food category", xlabel="Protein (g per 100 g food)", ylabel="USDA-derived category")
    ax.set_xlim(0, ordered["mean_protein"].max() * 1.2)
    path = directory / "02_protein_by_category.png"
    fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    matrix = np.array(model_result["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Blues")
    labels = model_result["labels"]
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, xlabel="Predicted category", ylabel="Actual category",
           title="Logistic Regression: held-out food families")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, label="Number of food descriptions", shrink=0.75)
    path = directory / "03_confusion_matrix.png"
    fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return paths
