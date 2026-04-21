import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

TOPIC_MAPPING = Path("../data/topic_category_mapping.csv")
FILTERING_AUDIT = Path("../data/filtering_audit_100.csv")

CATEGORY_NAMES = {
    1: "Technical safety",
    2: "Privacy and data misuse",
    3: "Fairness and discrimination",
    4: "Malicious misuse",
    5: "Societal and democratic risks",
}


df = pd.read_csv(TOPIC_MAPPING)

if {"Label_A", "Label_B"}.issubset(df.columns):
    kappa = cohen_kappa_score(df["Label_A"], df["Label_B"])
    print(f"Topic-to-category coder agreement (Cohen's kappa): {kappa:.3f}")
    print(f"Non-outlier topics coded: {len(df)}\n")

total_docs = df["Count"].sum()
agg = df.groupby("Gold_label").agg(
    n_topics=("Topic", "count"),
    n_documents=("Count", "sum"),
).reset_index()
agg["Category"] = agg["Gold_label"].map(CATEGORY_NAMES)
agg["Share (%)"] = (agg["n_documents"] / total_docs * 100).round(2)
agg = agg[["Gold_label", "Category", "n_topics", "n_documents", "Share (%)"]]

print("Category distribution over the structured non-outlier subset")
print("-" * 70)
print(agg.to_string(index=False))
print(f"\n  Total non-outlier documents: {total_docs:,}")

print("\nRepresentative topics per category (top 3 by document count)")
print("-" * 70)
for cat_id in sorted(df["Gold_label"].unique()):
    cat_name = CATEGORY_NAMES[cat_id]
    top = (
        df[df["Gold_label"] == cat_id]
        .sort_values("Count", ascending=False)
        .head(3)
    )
    print(f"\n  [{cat_id}] {cat_name}")
    for _, row in top.iterrows():
        print(f"      - {row['Topic name']} (n = {row['Count']})")


print("\n" + "=" * 70)
print("Keyword-prescreening coverage audit (n = 100 excluded tweets)")
print("=" * 70)
audit = pd.read_csv(FILTERING_AUDIT)
n = len(audit)
risk = int((audit["risk_label"] == 1).sum())
p = risk / n
lo, hi = proportion_confint(risk, n, alpha=0.05, method="wilson")
print(f"  Tweets coded as ethical risk discourse: {risk} / {n} ({p * 100:.1f}%)")
print(f"  Wilson 95% CI: [{lo * 100:.1f}%, {hi * 100:.1f}%]")
