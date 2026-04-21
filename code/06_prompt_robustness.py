import os
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score,
)

CSV_PATH   = Path("../data/human_annotation_385.csv")
OUTPUT_CSV = Path("../data/prompt_robustness_results.csv")

MODEL_NAME  = "gpt-4.1"
TEMPERATURE = 0.0
SAVE_EVERY  = 25

GOLD_COL      = "Gold_label"
REFERENCE_COL = "gpt_4_1_label"
TEXT_COL      = "content"

PROMPT_VARIANTS = [
    ("variant_a_minimal",     "../prompts/variant_a_minimal.txt"),
    ("variant_b_paraphrased", "../prompts/variant_b_paraphrased.txt"),
    ("variant_c_reordered",   "../prompts/variant_c_reordered.txt"),
]
PRIMARY_PROMPT_PATH = "../prompts/primary_prompt.txt"

INTER_RUN_COLS = ["run2", "run3"]

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Set OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

prompt_templates = {}
for name, path in PROMPT_VARIANTS:
    with open(path) as f:
        prompt_templates[name] = f.read()
with open(PRIMARY_PROMPT_PATH) as f:
    REFERENCE_PROMPT = f.read()


def classify_with_prompt(tweet, prompt_template):
    prompt = prompt_template.replace("{tweet}", tweet)
    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    label = response.output_text.strip()
    if label not in {"0", "1"}:
        raise ValueError(f"Non-binary output: {label!r}")
    return int(label)


def evaluate(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "Kappa":     cohen_kappa_score(y_true, y_pred),
    }


def run_column(df, col, prompt_template, out_path):
    if col not in df.columns:
        df[col] = pd.NA

    for i in range(len(df)):
        if pd.notna(df.at[i, col]):
            continue
        retry = 0
        while True:
            try:
                df.at[i, col] = classify_with_prompt(df.at[i, TEXT_COL], prompt_template)
                break
            except (RateLimitError, APIError, APITimeoutError):
                retry += 1
                wait = min(5 * (2 ** retry), 60)
                print(f"  [{col}] API error at row {i}, retry {retry}, wait {wait}s")
                time.sleep(wait)
            except Exception as e:
                print(f"  [{col}] Error at row {i}: {e}")
                df.at[i, col] = pd.NA
                break
        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(out_path, index=False)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    if TEXT_COL not in df.columns:
        raise RuntimeError(
            "human_annotation_385.csv must contain a 'content' column. "
            "Rehydrate tweet text from the Kaggle source (Ansari, 2023) first — "
            "see README section 'Data rehydration'."
        )

    print("=" * 60)
    print("Part 1: Prompt Robustness")
    print("=" * 60)

    for variant_col, _ in PROMPT_VARIANTS:
        print(f"\nRunning {variant_col}...")
        df = run_column(df, variant_col, prompt_templates[variant_col], OUTPUT_CSV)

    print("\nPrompt robustness results:")
    eval_cols = [REFERENCE_COL] + [v for v, _ in PROMPT_VARIANTS]
    for col in eval_cols:
        tmp = df[[GOLD_COL, col]].dropna()
        m = evaluate(tmp[GOLD_COL].astype(int), tmp[col].astype(int))
        if col == REFERENCE_COL:
            agree = 1.0
        else:
            ref_tmp = df[[REFERENCE_COL, col]].dropna()
            agree = (ref_tmp[REFERENCE_COL].astype(int) == ref_tmp[col].astype(int)).mean()
        print(f"  {col:26s}  Acc={m['Accuracy']:.3f}  "
              f"F1={m['F1']:.3f}  Kappa={m['Kappa']:.3f}  Agreement={agree:.3f}")

    print("\n" + "=" * 60)
    print("Part 2: Inter-Run Consistency")
    print("=" * 60)

    for run_col in INTER_RUN_COLS:
        print(f"\nRunning {run_col}...")
        df = run_column(df, run_col, REFERENCE_PROMPT, OUTPUT_CSV)

    print("\nPairwise agreement:")
    all_runs = [REFERENCE_COL] + INTER_RUN_COLS
    for i in range(len(all_runs)):
        for j in range(i + 1, len(all_runs)):
            a, b = all_runs[i], all_runs[j]
            tmp = df[[a, b]].dropna()
            ya, yb = tmp[a].astype(int), tmp[b].astype(int)
            agree = (ya == yb).mean()
            kappa = cohen_kappa_score(ya, yb)
            changed = (ya != yb).sum()
            print(f"  {a:26s} vs {b:26s}  "
                  f"Agreement={agree:.3f}  Changed={changed}  Kappa={kappa:.3f}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
