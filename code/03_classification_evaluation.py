import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix,
)

HUMAN_ANNOTATION = Path("../data/human_annotation_385.csv")
GPT4_FULL  = Path("../data/gpt_4_1_full_labels.csv")
GPT35_FULL = Path("../data/gpt_3_5_turbo_full_labels.csv")

GOLD_COL = "Gold_label"

MODEL_COLS = [
    ("GPT-3.5-turbo",     "gpt_3_5_turbo_label"),
    ("Claude Sonnet 4.6", "claude_sonnet_4_6_label"),
    ("Gemini 2.5 Pro",    "gemini_2_5_pro_label"),
    ("GPT-4.1",           "gpt_4_1_label"),
]


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "Kappa":     cohen_kappa_score(y_true, y_pred),
    }


df = pd.read_csv(HUMAN_ANNOTATION)
print(f"Loaded human annotation set: {len(df):,} rows")
print(f"Positive (risk) samples in Gold_label: "
      f"{(df[GOLD_COL] == 1).sum()}/{len(df)} "
      f"({(df[GOLD_COL] == 1).mean() * 100:.1f}%)\n")

if "coder_A_label" in df.columns and "coder_B_label" in df.columns:
    kappa = cohen_kappa_score(df["coder_A_label"], df["coder_B_label"])
    print(f"Inter-coder agreement (coder A vs coder B), Cohen's kappa: {kappa:.4f}\n")

print("=" * 70)
print("Per-model classification metrics (human annotation set, n = 385)")
print("=" * 70)

rows = []
for name, col in MODEL_COLS:
    if col not in df.columns:
        print(f"  [SKIP] column '{col}' not found; run 02_llm_classification.py first.")
        continue
    y_true = df[GOLD_COL]
    y_pred = df[col]
    metrics = compute_metrics(y_true, y_pred)
    rows.append({"Model": name, **metrics})

metrics_df = pd.DataFrame(rows)
print(metrics_df.round(3).to_string(index=False))

kw_y_true = df[GOLD_COL]
kw_y_pred = [1] * len(df)
kw_metrics = compute_metrics(kw_y_true, kw_y_pred)
print("\nKeyword baseline:")
print("  " + "  ".join(f"{k}={v:.3f}" for k, v in kw_metrics.items()))

print("\n" + "=" * 70)
print("Normalized confusion matrices (rows = actual, cols = predicted)")
print("=" * 70)
for name, col in MODEL_COLS:
    if col not in df.columns:
        continue
    cm = confusion_matrix(df[GOLD_COL], df[col], labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    print(f"\n  {name}")
    print(f"                 Pred=Non-Risk   Pred=Risk")
    print(f"    Actual=Non    {cm_norm[0,0]*100:6.1f}%         {cm_norm[0,1]*100:6.1f}%     (n={cm[0,0]}, {cm[0,1]})")
    print(f"    Actual=Risk   {cm_norm[1,0]*100:6.1f}%         {cm_norm[1,1]*100:6.1f}%     (n={cm[1,0]}, {cm[1,1]})")

print("\n" + "=" * 70)
print("Full candidate corpus (n = 48,398): assigned-label distribution")
print("=" * 70)

for path, col, display in [
    (GPT4_FULL,  "gpt_4_1_label",       "GPT-4.1"),
    (GPT35_FULL, "gpt_3_5_turbo_label", "GPT-3.5-turbo"),
]:
    if not path.exists():
        print(f"  [SKIP] {path} not found.")
        continue
    full_df = pd.read_csv(path)
    counts = full_df[col].value_counts(dropna=False).to_dict()
    risk = counts.get(1, 0)
    non_risk = counts.get(0, 0)
    total = risk + non_risk
    print(f"\n  {display}")
    print(f"    Risk     : {risk:>6,} ({risk/total*100:.2f}%)")
    print(f"    Non-Risk : {non_risk:>6,} ({non_risk/total*100:.2f}%)")
    print(f"    Total    : {total:>6,}")
