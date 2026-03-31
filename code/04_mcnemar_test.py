"""
04_mcnemar_test.py
Stage 1-D: McNemar's exact test comparing GPT-4.1 and GPT-3.5-turbo error distributions.

Input:  data/gold_validation_385.csv
Output: Console output (contingency table, test statistic, p-value)
"""

import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv("../data/gold_validation_385.csv")

y_true = df["Gold_label"]
y_gpt4 = df["gpt_label"]
y_gpt35 = df["gpt_3_5_label"]

# ── Correctness vectors ────────────────────────────────────
gpt4_correct = (y_gpt4 == y_true)
gpt35_correct = (y_gpt35 == y_true)

# ── 2×2 contingency table ─────────────────────────────────
a = int(np.sum(gpt4_correct & gpt35_correct))          # Both correct
b = int(np.sum(gpt4_correct & ~gpt35_correct))         # GPT-4.1 correct, GPT-3.5 incorrect
c = int(np.sum(~gpt4_correct & gpt35_correct))         # GPT-4.1 incorrect, GPT-3.5 correct
d = int(np.sum(~gpt4_correct & ~gpt35_correct))        # Both incorrect

table = [[a, b], [c, d]]

print("=== Table IV: McNemar Contingency Table ===")
print(f"                    GPT-3.5 Correct    GPT-3.5 Incorrect")
print(f"  GPT-4.1 Correct       {a:>5}              {b:>5}")
print(f"  GPT-4.1 Incorrect     {c:>5}              {d:>5}")
print()

# ── McNemar's exact test ───────────────────────────────────
result = mcnemar(table, exact=True)

print(f"Statistic (min of discordant pairs): {result.statistic}")
print(f"p-value:  {result.pvalue:.6f}")
print()

if result.pvalue < 0.05:
    print("Result: The difference in error distributions is statistically significant (p < 0.05).")
else:
    print("Result: No statistically significant difference in error distributions (p >= 0.05).")
