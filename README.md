# A Reproducible Framework for LLM-Based Classification and Structural Analysis of Ethical Risk Discourse in Social Media

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19570768.svg)](https://doi.org/10.5281/zenodo.19570768)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Replication materials for:

> Kim, S., Kim, S. H., & Lee, B. G. (2026). *A Reproducible Framework for LLM-Based Classification and Structural Analysis of Ethical Risk Discourse in Social Media.* PeerJ Computer Science (under review).

## Overview

This repository contains the code, prompts, tweet IDs, classification labels, and topic-to-category mapping results that reproduce the results reported in the paper. The framework consists of two stages.

**Stage 1 — LLM-based contextual classification.** Keyword-prescreened tweets are classified as ethical risk discourse or not using four large language models (GPT-4.1, GPT-3.5-turbo, Claude Sonnet 4.6, Gemini 2.5 Pro) under a zero-shot instruction prompt grounded in five predefined ethical risk categories. Model performance is evaluated against a human annotation set (n = 385) and compared with supervised baselines (TF-IDF + Logistic Regression, BERTweet, RoBERTa) trained via stratified five-fold cross-validation. GPT-4.1 was selected as the final classification model based on F1-score and Cohen's kappa, and applied to the full candidate corpus of 48,398 tweets.

**Stage 2 — BERTopic-based structural analysis.** The corpus classified as risk discourse by GPT-4.1 is analyzed using BERTopic, yielding 33 sub-topics after outlier exclusion. The resulting topics are systematically linked to the five higher-level ethical risk categories through an explicit topic-to-category mapping protocol documented in `mapping_guideline.md`.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── mapping_guideline.md                    # topic-to-category mapping guideline (paper Appendix B)
├── code/
│   ├── 01_keyword_prescreening.py          # Stage 1-A: keyword-based pre-screening
│   ├── 02_llm_classification.py            # Stage 1-B: four-LLM zero-shot classification
│   ├── 03_classification_evaluation.py     # Stage 1-C: per-model metrics vs. human annotation set
│   ├── 04_supervised_baselines.py          # Stage 1-D: TF-IDF+LR / BERTweet / RoBERTa (5-fold CV)
│   ├── 05_mcnemar_test.py                  # Stage 1-E: McNemar's exact test (six pairs)
│   ├── 06_prompt_robustness.py             # Stage 1-F: prompt variants and inter-run consistency
│   ├── 07_bertopic_analysis.py             # Stage 2-A: BERTopic modeling with sensitivity analysis
│   ├── 08_bertopic_seed_robustness.py      # Stage 2-A (supp.): UMAP seed robustness check
│   └── 09_category_aggregation.py          # Stage 2-B: category-level aggregation
├── data/
│   ├── candidate_corpus.csv                # 48,398 tweet IDs after keyword pre-screening
│   ├── human_annotation_385.csv            # 385 IDs with human and four LLM labels
│   ├── gpt_4_1_full_labels.csv             # 48,398 IDs with GPT-4.1 labels
│   ├── gpt_3_5_turbo_full_labels.csv       # 48,398 IDs with GPT-3.5-turbo labels
│   ├── filtering_audit_100.csv             # 100 excluded tweets, coded for risk presence
│   ├── topic_category_mapping.csv          # 33 topics with two independent coders + reconciled label
│   ├── supervised_baseline_performance.csv # out-of-fold metrics for supervised baselines
│   ├── supervised_baseline_oof_predictions.csv  # per-tweet out-of-fold predictions
│   └── mcnemar_pairwise_results.csv        # six pairwise McNemar results
└── prompts/
    ├── primary_prompt.txt                  # primary classification prompt (paper Appendix A.2)
    ├── variant_a_minimal.txt               # prompt robustness Variant A
    ├── variant_b_paraphrased.txt           # prompt robustness Variant B
    └── variant_c_reordered.txt             # prompt robustness Variant C
```

## Data

### Source dataset

The ~500,000 ChatGPT-related tweets (January 4 – March 29, 2023) are from:

> Ansari, K. (2023). *500k ChatGPT-related Tweets Jan–Mar 2023.* Kaggle. https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023

License of source: CC0 1.0 Public Domain.

### Data rehydration

In accordance with platform terms and data-sharing restrictions, this repository does **not** include raw tweet text. Only tweet IDs and derived annotations are provided, to allow rehydration where permitted.

To reconstruct a working corpus with tweet text (the `content` column required by several scripts), obtain the source text from one of the following and join on the `id` column:

1. **Kaggle (recommended).** Download the CSV from the Ansari (2023) dataset above and place it at `data/Twitter_Jan_Mar.csv`. This CSV already contains the `content` column for all ~500,000 tweets, so a simple `pandas.merge` on `id` restores the text for the IDs in `candidate_corpus.csv`, `human_annotation_385.csv`, and the full-corpus label files.
2. **X (Twitter) API.** Rehydrate tweet text by tweet ID via the X API, subject to the platform's terms of service and the access level available to you.

Users who only need to reproduce downstream analyses (evaluation, McNemar's test, category aggregation, filtering audit) can do so without rehydration, since those scripts do not require `content`.

### Provided data files

| File | Rows | Description |
|---|---:|---|
| `candidate_corpus.csv` | 48,398 | Tweet IDs after keyword-based pre-screening (`date`, `id`, `matched_keywords`) |
| `human_annotation_385.csv` | 385 | Validation sample with independent human labels (`coder_A_label`, `coder_B_label`), reconciled `Gold_label`, and four LLM labels (`gpt_4_1_label`, `gpt_3_5_turbo_label`, `claude_sonnet_4_6_label`, `gemini_2_5_pro_label`) |
| `gpt_4_1_full_labels.csv` | 48,398 | Full candidate corpus with GPT-4.1 zero-shot labels |
| `gpt_3_5_turbo_full_labels.csv` | 48,398 | Full candidate corpus with GPT-3.5-turbo zero-shot labels |
| `filtering_audit_100.csv` | 100 | Random sample from pre-screening-excluded tweets, coded for risk presence |
| `topic_category_mapping.csv` | 33 | Non-outlier BERTopic topics with two independent coders' labels (`Label_A`, `Label_B`), reconciled `Gold_label`, topic name, document count, and top keywords |
| `supervised_baseline_performance.csv` | 5 | Out-of-fold metrics for TF-IDF+LR, BERTweet, RoBERTa, GPT-3.5-turbo, GPT-4.1 |
| `supervised_baseline_oof_predictions.csv` | 385 | Per-tweet out-of-fold predictions from the five models |
| `mcnemar_pairwise_results.csv` | 6 | Pairwise McNemar exact-test results for the six pairs reported in the paper |

Cross-LLM comparison was performed **only on the human annotation set (n = 385)**. On the full candidate corpus (n = 48,398), only GPT-4.1 and GPT-3.5-turbo were applied, consistent with the paper's design.

## Ethical risk categories

Five higher-level ethical risk categories are used throughout both stages (paper Methodology, "Task Definition and Label Schema"):

1. **Technical safety** — risks related to system reliability and safety, including hallucinations, harmful content generation, and security vulnerabilities.
2. **Privacy and data misuse** — risks associated with privacy and data governance, including personal data leakage and unauthorized data use.
3. **Fairness and discrimination** — risks related to fairness and rights, including algorithmic bias, discriminatory outcomes, and violations of creators' rights such as copyright infringement.
4. **Malicious misuse** — risks involving intentional harmful use, including deepfakes, misinformation, jailbreak exploitation, and criminal applications.
5. **Societal and democratic risks** — structural risks affecting society, including labor displacement, threats to democracy, the digital divide, human rights concerns, and environmental impacts.

Conversely, general usage experiences, praise, humor, product comparisons, feature descriptions, or neutral mentions without risk implications are labeled 0.

## Pipeline

### Stage 1 — LLM-based risk discourse classification

| Step | Script | Purpose |
|---|---|---|
| 1-A | `01_keyword_prescreening.py` | Apply keyword-based pre-screening to the ~500K raw tweets; produce the 48,398-tweet candidate corpus. |
| 1-B | `02_llm_classification.py` | Classify tweets using four LLMs (OpenAI GPT-4.1, GPT-3.5-turbo; Anthropic Claude Sonnet 4.6; Google Gemini 2.5 Pro). Full corpus classification uses only GPT-4.1 and GPT-3.5-turbo; the human annotation set (n = 385) is classified by all four models for the cross-model comparison. |
| 1-C | `03_classification_evaluation.py` | Compute accuracy, precision, recall, F1, and Cohen's kappa against `Gold_label`; report normalized confusion matrices and the full-corpus risk distribution for GPT-4.1 and GPT-3.5-turbo. |
| 1-D | `04_supervised_baselines.py` | Train TF-IDF + Logistic Regression, BERTweet, and RoBERTa on the human annotation set via stratified 5-fold cross-validation and compare with GPT-3.5-turbo and GPT-4.1 on the same folds. |
| 1-E | `05_mcnemar_test.py` | McNemar's exact test for the six model pairs reported in the paper. |
| 1-F | `06_prompt_robustness.py` | Evaluate GPT-4.1 under three prompt variants and assess inter-run consistency across three repeated runs. |

### Stage 2 — BERTopic-based structural analysis

| Step | Script | Purpose |
|---|---|---|
| 2-A | `07_bertopic_analysis.py` | Text preprocessing, BERTopic modeling, sensitivity analysis over `min_cluster_size ∈ {50, 60, 70, 80, 90, 100}`, and final model fitting at `min_cluster_size = 90`. |
| 2-A (supp.) | `08_bertopic_seed_robustness.py` | Re-run the final configuration with five UMAP seeds randomly drawn from values other than the primary seed (42) to assess seed dependence. |
| 2-B | `09_category_aggregation.py` | Aggregate the 33 non-outlier topics into the five ethical risk categories using `topic_category_mapping.csv`; report category-level shares and the keyword-prescreening audit (Wilson 95% CI). |

## Installation

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### API keys

Scripts that call LLM APIs require the corresponding environment variables:

```bash
export OPENAI_API_KEY="..."       # required by 02_llm_classification.py, 06_prompt_robustness.py
export ANTHROPIC_API_KEY="..."    # required by 02_llm_classification.py (Claude Sonnet 4.6)
export GOOGLE_API_KEY="..."       # required by 02_llm_classification.py (Gemini 2.5 Pro)
```

### Running the pipeline

Scripts are designed to run sequentially from the `code/` directory. Stages that only read the provided label files (`03`, `05`, `09`) can be run without API access or rehydrated tweet text.

```bash
cd code

# Stage 1
python 01_keyword_prescreening.py            # requires rehydrated data/Twitter_Jan_Mar.csv
python 02_llm_classification.py              # requires all three API keys; runs both full + validation scopes
python 02_llm_classification.py --scope validation    # only the 385-sample cross-LLM comparison
python 02_llm_classification.py --scope full          # only the full-corpus classification (GPT-4.1, GPT-3.5-turbo)
python 03_classification_evaluation.py       # runs on provided label CSVs; no API needed
python 04_supervised_baselines.py            # requires content column; GPU recommended for BERTweet/RoBERTa
python 05_mcnemar_test.py                    # runs on provided CSVs; no API needed
python 06_prompt_robustness.py               # requires OPENAI_API_KEY

# Stage 2
python 07_bertopic_analysis.py               # requires content column
python 08_bertopic_seed_robustness.py        # requires content column; supplementary
python 09_category_aggregation.py            # runs on provided CSVs; no API needed
```

Scripts `02`, `04`, `06`, `07`, and `08` make many API calls or require GPU fine-tuning and may take from tens of minutes to several hours. Pre-computed label CSVs are provided under `data/` so that downstream analyses can be reproduced without re-running these stages.

### BERTopic seed robustness

`08_bertopic_seed_robustness.py` re-runs the final BERTopic configuration (`min_cluster_size = 90`) with five UMAP seeds randomly sampled (per execution) from values other than the primary seed (`random_state = 42`) used in `07_bertopic_analysis.py`. Coherence (`C_v`, `C_NPMI`), topic count, outlier ratio, and topic diversity are reported as cross-seed mean ± standard deviation, together with per-seed topic keywords.

The coherence calculation reuses the same tokenization pipeline (CountVectorizer analyzer with `ngram_range = (1, 2)`) as `07_bertopic_analysis.py`, so seed-robustness values are directly comparable to the sensitivity analysis reported in the paper. Because seeds are drawn at runtime, exact seed values used in the paper may not be reproduced; the substantive claim — that seed dependence is limited — is expected to hold across runs.

## Key parameters

| Component | Parameter | Value |
|---|---|---|
| LLM classification | Temperature | 0.0 |
| Sentence embedding | Model | `all-mpnet-base-v2` |
| UMAP | n_neighbors / n_components / min_dist / metric | 15 / 5 / 0.0 / cosine |
| HDBSCAN | min_cluster_size (final) | 90 |
| HDBSCAN | min_samples / metric | 10 / euclidean |
| c-TF-IDF | BM25 weighting + reduce_frequent_words | enabled |
| CountVectorizer | ngram_range | (1, 2) |
| Supervised baselines | CV folds / max_len / batch / epochs / LR | 5 / 128 / 16 / 3 / 2e-5 |

## Citation

```bibtex
@article{kim2026reproducible,
  title   = {A Reproducible Framework for LLM-Based Classification and Structural Analysis of Ethical Risk Discourse in Social Media},
  author  = {Kim, Soyon and Kim, Soo Hyung and Lee, Bong Gyou},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under review}
}
```

Archived release: https://doi.org/10.5281/zenodo.19570768

## License

Code in this repository is released under the [MIT License](LICENSE). The source tweet dataset (Ansari, 2023) is released under CC0 1.0 Public Domain.
