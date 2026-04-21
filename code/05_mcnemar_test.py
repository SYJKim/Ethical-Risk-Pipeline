import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

OOF_PATH = Path("../data/supervised_baseline_oof_predictions.csv")
OUT_PATH = Path("../data/mcnemar_pairwise_results.csv")

MODEL_COLS = {
    "TF-IDF + LR": "tfidf_lr_pred",
    "BERTweet":    "bertweet_pred",
    "RoBERTa":     "roberta_pred",
    "GPT-4.1":     "gpt4_pred",
}

PAIRS = [
    ("TF-IDF + LR", "BERTweet"),
    ("TF-IDF + LR", "RoBERTa"),
    ("TF-IDF + LR", "GPT-4.1"),
    ("BERTweet",    "RoBERTa"),
    ("BERTweet",    "GPT-4.1"),
    ("RoBERTa",     "GPT-4.1"),
]


def significance_marker(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


df = pd.read_csv(OOF_PATH)
y_true = df["Gold_label"].astype(int).values

rows = []
for name_a, name_b in PAIRS:
    pred_a = df[MODEL_COLS[name_a]].astype(int).values
    pred_b = df[MODEL_COLS[name_b]].astype(int).values

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_correct = int(np.sum(correct_a & correct_b))
    both_wrong   = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=True)
    p = float(result.pvalue)

    rows.append({
        "Model A":        name_a,
        "Model B":        name_b,
        "A_only_correct": a_only,
        "B_only_correct": b_only,
        "p_value":        round(p, 4),
        "Sig.":           significance_marker(p),
    })

out = pd.DataFrame(rows)
print("McNemar's exact test — pairwise comparisons (n = 385)")
print(out.to_string(index=False))
print("\nSignificance: * p < .05, ** p < .01, *** p < .001")

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
