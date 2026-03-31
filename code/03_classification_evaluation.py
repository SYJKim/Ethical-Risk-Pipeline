"""
03_classification_evaluation.py
Stage 1-C: Evaluate LLM classification against human-annotated gold labels.

Input:  data/gold_validation_385.csv
Output: Console output (metrics table, confusion matrices)
        figures/ (performance comparison, confusion matrices, error decomposition, full-corpus distribution)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix,
)

# ── Configuration ──────────────────────────────────────────
CSV_PATH = Path("../data/gold_validation_385.csv")
OUT_DIR = Path("../figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLD_COL = "Gold_label"
MODEL_COLS = {
    "GPT-4.1": "gpt_label",
    "GPT-3.5-turbo": "gpt_3_5_label",
}

# Full corpus counts (from 48,398 candidate tweets)
FULL_CORPUS_COUNTS = {
    "GPT-4.1":       {"Risk": 15802, "Non-Risk": 32596},
    "GPT-3.5-turbo": {"Risk": 12473, "Non-Risk": 35925},
}


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-score":  f1_score(y_true, y_pred, zero_division=0),
        "Cohen's κ": cohen_kappa_score(y_true, y_pred),
        "CM":        cm,
    }


def save_figure(fig, filename_stem):
    fig.savefig(OUT_DIR / f"{filename_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{filename_stem}.pdf", bbox_inches="tight")
    print(f"Saved: {OUT_DIR / filename_stem}.[png|pdf]")


# ── 1. Load data and compute metrics ──────────────────────
df = pd.read_csv(CSV_PATH)
results = {}
for name, col in MODEL_COLS.items():
    results[name] = compute_metrics(df[GOLD_COL], df[col])

# Print metrics table
rows = []
for name, vals in results.items():
    rows.append({
        "Model": name,
        "Accuracy": vals["Accuracy"],
        "Precision": vals["Precision"],
        "Recall": vals["Recall"],
        "F1-score": vals["F1-score"],
        "Cohen's κ": vals["Cohen's κ"],
    })
metrics_df = pd.DataFrame(rows)
print("\n=== Table III: Model Performance Comparison ===")
print(metrics_df.round(3).to_string(index=False))

# Add keyword baseline
print("\n  Keyword baseline: Accuracy=0.364, Precision=0.364, Recall=1.000, F1=0.533, κ=0.000")

# ── 2. Performance comparison (slope chart) ───────────────
metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "Cohen's κ"]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
x = np.array([0, 1])

for metric in metric_names:
    y = [results["GPT-3.5-turbo"][metric], results["GPT-4.1"][metric]]
    line, = ax.plot(x, y, marker="o", linewidth=2)
    ax.text(1.03, y[1], f"{metric} ({y[1]:.3f})", color=line.get_color(),
            fontsize=10, va="center", ha="left")

ax.set_xlim(-0.02, 1.32)
ax.set_ylim(0.55, 0.95)
ax.set_xticks(x)
ax.set_xticklabels(["GPT-3.5-turbo", "GPT-4.1"])
ax.set_ylabel("Score")
ax.set_title("Figure 2. Model Performance Comparison")
ax.grid(axis="y", alpha=0.3)
save_figure(fig, "figure2_performance_comparison")
plt.close()

# ── 3. Normalized confusion matrices ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
for ax, name in zip(axes, ["GPT-4.1", "GPT-3.5-turbo"]):
    cm = results[name]["CM"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNon-Risk", "Predicted\nRisk"])
    ax.set_yticklabels(["Actual\nNon-Risk", "Actual\nRisk"])
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i,j]*100:.1f}%\n(n={cm[i,j]})",
                    ha="center", va="center", fontsize=10)

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="Row-normalized proportion")
fig.suptitle("Figure 3. Normalized Confusion Matrices", fontsize=13)
save_figure(fig, "figure3_confusion_matrices")
plt.close()

# ── 4. Full-corpus risk distribution ─────────────────────
full_df = pd.DataFrame(FULL_CORPUS_COUNTS).T.reset_index().rename(columns={"index": "Model"})
full_df["Total"] = full_df["Risk"] + full_df["Non-Risk"]
full_df["Risk_pct"] = full_df["Risk"] / full_df["Total"]
full_df["NonRisk_pct"] = full_df["Non-Risk"] / full_df["Total"]

fig, ax = plt.subplots(figsize=(7, 4.5))
y = np.arange(len(full_df))
ax.barh(y, full_df["Risk_pct"], label="Risk")
ax.barh(y, full_df["NonRisk_pct"], left=full_df["Risk_pct"], label="Non-Risk")
ax.set_yticks(y)
ax.set_yticklabels(full_df["Model"])
ax.set_xlim(0, 1)
ax.set_xlabel("Proportion")
ax.set_title("Figure 5. Model-Assigned Risk Distribution on the Full Corpus")
ax.legend(frameon=False)
for i, row in full_df.iterrows():
    ax.text(row["Risk_pct"]/2, i, f'Risk\n{row["Risk"]:,} ({row["Risk_pct"]*100:.2f}%)',
            ha="center", va="center", fontsize=9)
    ax.text(row["Risk_pct"] + row["NonRisk_pct"]/2, i,
            f'Non-Risk\n{row["Non-Risk"]:,} ({row["NonRisk_pct"]*100:.2f}%)',
            ha="center", va="center", fontsize=9)
save_figure(fig, "figure5_full_corpus_distribution")
plt.close()

# ── 5. Save summary CSVs ─────────────────────────────────
metrics_df.to_csv(OUT_DIR / "validation_metrics_summary.csv", index=False)
full_df.to_csv(OUT_DIR / "full_corpus_distribution.csv", index=False)
print(f"\nAll outputs saved in: {OUT_DIR.resolve()}")
