"""
02_llm_classification.py
Stage 1-B: LLM-based contextual classification using GPT-4.1 and GPT-3.5-turbo.

Input:  data/candidate_corpus.csv
Output: data/gpt4_1_full_labels.csv
        data/gpt3_5_full_labels.csv

Requires: OPENAI_API_KEY environment variable.
"""

import os
import time
import random
import pandas as pd
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

# ── Configuration ──────────────────────────────────────────
INPUT_PATH = Path("../data/candidate_corpus.csv")
OUTPUT_DIR = Path("../data")
PROMPT_PATH = Path("../prompts/primary_prompt.txt")

MODELS = {
    "gpt-4.1":       "gpt4_1_full_labels.csv",
    "gpt-3.5-turbo": "gpt3_5_full_labels.csv",
}

TEMPERATURE = 0.0
SAVE_EVERY = 1000
MAX_RETRIES = 10
BASE_WAIT = 5
MAX_WAIT = 300
SUCCESS_SLEEP = 1.2

# ── Setup ──────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Set OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

with open(PROMPT_PATH) as f:
    PROMPT_TEMPLATE = f.read()


def classify(tweet: str, model: str) -> str:
    """Classify a single tweet using the specified model."""
    prompt = PROMPT_TEMPLATE.replace("{tweet}", tweet)
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return response.output_text.strip()


def run_classification(df: pd.DataFrame, model_name: str, output_path: Path):
    """Classify all tweets and save results with resume support."""
    label_col = "gpt_label"

    # Resume: load existing progress if available
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"  Resuming from {output_path} ({df[label_col].notna().sum():,} already labeled)")
    elif label_col not in df.columns:
        df[label_col] = None

    total = len(df)
    print(f"  Total tweets: {total:,}")

    for i in range(total):
        if pd.notna(df.at[i, label_col]):
            continue

        retries = 0
        while True:
            try:
                label = classify(df.iloc[i]["content"], model_name)
                if label not in {"0", "1"}:
                    raise ValueError(f"Non-binary output: {label}")
                df.at[i, label_col] = int(label)
                time.sleep(SUCCESS_SLEEP)
                break

            except (RateLimitError, APIError, APITimeoutError):
                retries += 1
                wait = min(BASE_WAIT * (2 ** retries), MAX_WAIT)
                wait *= 0.7 + 0.6 * random.random()
                print(f"  API error at row {i} | retry {retries}/{MAX_RETRIES} | wait {wait:.1f}s")
                time.sleep(wait)
                if retries >= MAX_RETRIES:
                    print(f"  [SKIP] Row {i} skipped after max retries.")
                    df.at[i, label_col] = None
                    break

            except Exception as e:
                print(f"  Unexpected error at row {i}: {e}")
                df.at[i, label_col] = None
                break

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(output_path, index=False)
            labeled = df[label_col].notna().sum()
            print(f"  Progress: {labeled:,}/{total:,} saved to {output_path}")

    df.to_csv(output_path, index=False)
    counts = df[label_col].value_counts(dropna=False)
    print(f"\n  Classification complete. Saved to {output_path}")
    print(f"  Label distribution:\n{counts.to_string()}\n")


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = pd.read_csv(INPUT_PATH)
    print(f"Loaded candidate corpus: {len(df_raw):,} tweets\n")

    for model_name, output_file in MODELS.items():
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        output_path = OUTPUT_DIR / output_file
        run_classification(df_raw.copy(), model_name, output_path)
