"""
05_prompt_robustness.py
Stage 1-E: Prompt robustness check and inter-run consistency check for GPT-4.1.

Input:  data/gold_validation_385.csv
Output: Console output (robustness metrics, inter-run agreement)

Requires: OPENAI_API_KEY environment variable.

Note: This script makes API calls. Pre-computed results are reported in the paper
      (Tables A4, A5). Running this script will reproduce those results but requires
      API access and may take ~1 hour.
"""

import os
import time
import textwrap
import pandas as pd
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score,
)

# ── Configuration ──────────────────────────────────────────
CSV_PATH = Path("../data/gold_validation_385.csv")
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.0
SAVE_EVERY = 25

GOLD_COL = "Gold_label"
REFERENCE_COL = "gpt_label"
TEXT_COL = "content"

# ── Setup ──────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Set OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)


def classify_with_prompt(tweet: str, prompt_template: str, model: str = MODEL_NAME) -> str:
    prompt = prompt_template.replace("{tweet}", tweet)
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    label = response.output_text.strip()
    if label not in {"0", "1"}:
        raise ValueError(f"Non-binary output: {label}")
    return label


def evaluate(y_true, y_pred):
    return {
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "F1-score":    f1_score(y_true, y_pred, zero_division=0),
        "Cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


# ── Load prompts ──────────────────────────────────────────
PROMPT_VARIANTS = {}
for name, path in [
    ("variant_a_minimal",      "../prompts/variant_a_minimal.txt"),
    ("variant_b_paraphrased",  "../prompts/variant_b_paraphrased.txt"),
    ("variant_c_reordered",    "../prompts/variant_c_reordered.txt"),
]:
    with open(path) as f:
        PROMPT_VARIANTS[name] = f.read()

with open("prompts/primary_prompt.txt") as f:
    REFERENCE_PROMPT = f.read()


def run_variant(df, variant_col, prompt_template, outfile):
    """Classify all 385 tweets with a prompt variant (resume-safe)."""
    if variant_col not in df.columns:
        df[variant_col] = pd.NA

    total = len(df)
    for i in range(total):
        if pd.notna(df.at[i, variant_col]):
            continue
        retry = 0
        while True:
            try:
                label = classify_with_prompt(df.at[i, TEXT_COL], prompt_template)
                df.at[i, variant_col] = int(label)
                break
            except (RateLimitError, APIError, APITimeoutError):
                retry += 1
                wait = min(5 * (2 ** retry), 60)
                print(f"  [{variant_col}] API error at row {i}, retry {retry}, wait {wait}s")
                time.sleep(wait)
            except Exception as e:
                print(f"  [{variant_col}] Error at row {i}: {e}")
                df.at[i, variant_col] = pd.NA
                break
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(outfile, index=False)
    df.to_csv(outfile, index=False)
    return df


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    outfile = "data/prompt_robustness_results.csv"

    # ── Part 1: Prompt robustness ─────────────────────────
    print("=" * 60)
    print("PART 1: Prompt Robustness Check")
    print("=" * 60)

    for variant_col, prompt_template in PROMPT_VARIANTS.items():
        print(f"\nRunning {variant_col}...")
        df = run_variant(df, variant_col, prompt_template, outfile)

    # Evaluate
    print("\n=== Prompt Robustness Results (Table A4) ===")
    eval_cols = [REFERENCE_COL] + list(PROMPT_VARIANTS.keys())
    for col in eval_cols:
        tmp = df[[GOLD_COL, col]].dropna()
        metrics = evaluate(tmp[GOLD_COL].astype(int), tmp[col].astype(int))

        if col == REFERENCE_COL:
            agree = 1.0
        else:
            ref_tmp = df[[REFERENCE_COL, col]].dropna()
            agree = (ref_tmp[REFERENCE_COL].astype(int) == ref_tmp[col].astype(int)).mean()

        print(f"  {col:30s}  Acc={metrics['Accuracy']:.3f}  F1={metrics['F1-score']:.3f}  "
              f"κ={metrics['Cohen_kappa']:.3f}  Agree_w_ref={agree:.3f}")

    # ── Part 2: Inter-run consistency ─────────────────────
    print("\n" + "=" * 60)
    print("PART 2: Inter-Run Consistency Check")
    print("=" * 60)

    INTER_RUN_COLS = ["run2", "run3"]
    for run_col in INTER_RUN_COLS:
        print(f"\nRunning {run_col}...")
        df = run_variant(df, run_col, REFERENCE_PROMPT, outfile)

    # Pairwise agreement
    print("\n=== Inter-Run Pairwise Agreement (Table A5) ===")
    all_runs = [REFERENCE_COL] + INTER_RUN_COLS
    for i in range(len(all_runs)):
        for j in range(i + 1, len(all_runs)):
            a, b = all_runs[i], all_runs[j]
            tmp = df[[a, b]].dropna()
            ya, yb = tmp[a].astype(int), tmp[b].astype(int)
            agree = (ya == yb).mean()
            kappa = cohen_kappa_score(ya, yb)
            print(f"  {a:20s} vs {b:20s}  Agreement={agree:.3f}  κ={kappa:.3f}")

    df.to_csv(outfile, index=False)
    print(f"\nAll results saved to {outfile}")
