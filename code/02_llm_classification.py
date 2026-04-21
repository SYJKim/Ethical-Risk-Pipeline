import os
import time
import random
import argparse
import pandas as pd
from pathlib import Path

# Paths and config
PROMPT_PATH = Path("../prompts/primary_prompt.txt")
CANDIDATE_CORPUS = Path("../data/candidate_corpus.csv")
HUMAN_ANNOTATION = Path("../data/human_annotation_385.csv")
DATA_DIR = Path("../data")

TEMPERATURE = 0.0
SAVE_EVERY = 1000
SAVE_EVERY_SMALL = 25
SUCCESS_SLEEP = 1.2
BASE_WAIT = 5
MAX_WAIT = 300
MAX_RETRIES = 10

with open(PROMPT_PATH) as f:
    PROMPT_TEMPLATE = f.read()


# Per-provider classifiers
def _openai_classify(tweet, model):
    from openai import OpenAI
    client = _openai_classify.client
    prompt = PROMPT_TEMPLATE.replace("{tweet}", tweet)
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return response.output_text.strip()


def _anthropic_classify(tweet, model):
    client = _anthropic_classify.client
    prompt = PROMPT_TEMPLATE.replace("{tweet}", tweet)
    response = client.messages.create(
        model=model,
        max_tokens=5,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _gemini_classify(tweet, model):
    from google.genai import types
    client = _gemini_classify.client
    prompt = PROMPT_TEMPLATE.replace("{tweet}", tweet)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    return (response.text or "").strip()


def get_classifier(provider, model_id):
    if provider == "openai":
        from openai import OpenAI
        _openai_classify.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return lambda tweet: _openai_classify(tweet, model_id)

    if provider == "anthropic":
        from anthropic import Anthropic
        _anthropic_classify.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return lambda tweet: _anthropic_classify(tweet, model_id)

    if provider == "google":
        from google import genai
        _gemini_classify.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        return lambda tweet: _gemini_classify(tweet, model_id)

    raise ValueError(f"Unknown provider: {provider}")


def get_retryable_exceptions(provider):
    if provider == "openai":
        from openai import RateLimitError, APIError, APITimeoutError
        return (RateLimitError, APIError, APITimeoutError)
    if provider == "anthropic":
        from anthropic import RateLimitError, APIError, APITimeoutError, APIConnectionError
        return (RateLimitError, APIError, APITimeoutError, APIConnectionError)
    if provider == "google":
        from google.genai import errors as genai_errors
        return (genai_errors.APIError,)
    return ()


def classify_corpus(df, text_col, label_col, classify_fn, retry_excs,
                    output_path, save_every):
    if output_path.exists():
        df = pd.read_csv(output_path)
        n_done = df[label_col].notna().sum() if label_col in df.columns else 0
        print(f"  Resuming from {output_path} ({n_done:,} already labeled)")
    elif label_col not in df.columns:
        df[label_col] = None

    total = len(df)
    print(f"  Total rows: {total:,}")

    for i in range(total):
        if pd.notna(df.at[i, label_col]):
            continue

        retries = 0
        while True:
            try:
                label = classify_fn(df.iloc[i][text_col])
                if label not in {"0", "1"}:
                    raise ValueError(f"Non-binary output: {label!r}")
                df.at[i, label_col] = int(label)
                time.sleep(SUCCESS_SLEEP)
                break

            except retry_excs as e:
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

        if (i + 1) % save_every == 0:
            df.to_csv(output_path, index=False)
            labeled = df[label_col].notna().sum()
            print(f"  Progress: {labeled:,}/{total:,} saved")

    df.to_csv(output_path, index=False)
    print(f"  Done. Saved to {output_path}")
    return df


FULL_CORPUS_MODELS = [
    ("openai",    "gpt-4.1",       "gpt_4_1_full_labels.csv",       "gpt_4_1_label"),
    ("openai",    "gpt-3.5-turbo", "gpt_3_5_turbo_full_labels.csv", "gpt_3_5_turbo_label"),
]

VALIDATION_MODELS = [
    ("openai",    "gpt-4.1",          "gpt_4_1_label"),
    ("openai",    "gpt-3.5-turbo",    "gpt_3_5_turbo_label"),
    ("anthropic", "claude-sonnet-4-6","claude_sonnet_4_6_label"),
    ("google",    "gemini-2.5-pro",   "gemini_2_5_pro_label"),
]


def run_full_corpus():
    df_raw = pd.read_csv(CANDIDATE_CORPUS)
    if "content" not in df_raw.columns:
        raise RuntimeError(
            "candidate_corpus.csv must contain a 'content' column. "
            "Rehydrate tweet text from the Kaggle source (Ansari, 2023) first — "
            "see README section 'Data rehydration'."
        )
    print(f"Loaded candidate corpus: {len(df_raw):,} tweets\n")

    for provider, model_id, outfile, label_col in FULL_CORPUS_MODELS:
        print("=" * 60)
        print(f"[Full corpus] Model: {model_id}")
        print("=" * 60)
        classify_fn = get_classifier(provider, model_id)
        retry_excs = get_retryable_exceptions(provider)
        classify_corpus(
            df_raw.copy(), "content", label_col, classify_fn, retry_excs,
            output_path=DATA_DIR / outfile, save_every=SAVE_EVERY,
        )


def run_validation_set():
    df = pd.read_csv(HUMAN_ANNOTATION)
    if "content" not in df.columns:
        raise RuntimeError(
            "human_annotation_385.csv must contain a 'content' column. "
            "Rehydrate tweet text from the Kaggle source (Ansari, 2023) first — "
            "see README section 'Data rehydration'."
        )
    print(f"Loaded human annotation set: {len(df):,} tweets\n")

    for provider, model_id, label_col in VALIDATION_MODELS:
        print("=" * 60)
        print(f"[Validation n=385] Model: {model_id} -> column: {label_col}")
        print("=" * 60)
        classify_fn = get_classifier(provider, model_id)
        retry_excs = get_retryable_exceptions(provider)
        classify_corpus(
            df, "content", label_col, classify_fn, retry_excs,
            output_path=HUMAN_ANNOTATION, save_every=SAVE_EVERY_SMALL,
        )
        df = pd.read_csv(HUMAN_ANNOTATION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based contextual classification")
    parser.add_argument(
        "--scope", choices=["full", "validation", "both"], default="both",
        help="'full': 48,398-tweet candidate corpus (GPT-4.1, GPT-3.5-turbo). "
             "'validation': 385-sample human annotation set (all four LLMs). "
             "'both': run both stages sequentially (default).",
    )
    args = parser.parse_args()

    if args.scope in ("full", "both"):
        run_full_corpus()
    if args.scope in ("validation", "both"):
        run_validation_set()
