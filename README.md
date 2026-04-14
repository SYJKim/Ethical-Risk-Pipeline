# A Reproducible and Human-Validated Pipeline for LLM-Based Classification of Ethical Risk Discourse in Large-Scale Social Media Data

## Research Overview

Isolating sparse, context-dependent target discourse from large-scale social media corpora remains a methodological challenge. This study presents a reproducible and human-validated pipeline that combines recall-oriented candidate construction, zero-shot LLM-based contextual classification, human validation, and BERTopic-based topic structure extraction. The pipeline was applied to approximately 500,000 ChatGPT-related tweets collected between January and March 2023 from Twitter (now X), representing the early phase of generative AI adoption. In a cross-model comparison, GPT-4.1 outperformed GPT-3.5-turbo across all evaluation metrics, achieving an accuracy of 0.899, an F1-score of 0.859, and a Cohen's κ of 0.780, with most of the improvement driven by a lower false-negative rate. BERTopic analysis of the pipeline-identified corpus yielded 33 sub-topics, which were grouped into five ethical risk categories through topic-to-category mapping. Within the structured non-outlier portion of the corpus, Societal and democratic risks accounted for the largest share (36.16%). The primary contribution is methodological: a human-validated modular workflow can isolate sparse, context-dependent target discourse from noisy social media data more reliably than keyword filtering alone and support interpretable downstream topic analysis.

This repository contains the replication materials for:

> Kim, S., Kim, S. H., & Lee, B. G. (2026). A Reproducible and Human-Validated Pipeline for LLM-Based Classification of Ethical Risk Discourse in Large-Scale Social Media Data. *PeerJ Computer Science*. (under review)

## Repository Structure

```
├── README.md
├── code/
│   ├── 01_keyword_prescreening.py          # Stage 1-A: Keyword-based pre-screening
│   ├── 02_llm_classification.py            # Stage 1-B: GPT-4.1 / GPT-3.5-turbo classification
│   ├── 03_classification_evaluation.py     # Stage 1-C: Validation metrics, confusion matrices, qualitative comparison
│   ├── 04_mcnemar_test.py                  # Stage 1-D: McNemar's exact test
│   ├── 05_prompt_robustness.py             # Stage 1-E: Prompt robustness & inter-run consistency
│   └── 06_bertopic_analysis.py             # Stage 2:   BERTopic modeling & sensitivity analysis
├── data/
│   ├── gold_validation_385.csv             # Gold validation set (385 tweets, human + model labels)
│   ├── gpt4_1_full_labels.csv              # GPT-4.1 labels on full candidate corpus (48,398 tweets)
│   ├── gpt3_5_full_labels.csv              # GPT-3.5-turbo labels on full candidate corpus
│   ├── filtering_audit_100.csv             # Audit of 100 excluded tweets
│   ├── topic_category_mapping.csv          # 33 topics → 5 ethical risk categories (coder A/B + Gold)
│   └── candidate_corpus.csv                # keyword-filtered candidate corpus
└── prompts/
    ├── primary_prompt.txt                  # Primary classification prompt (Table A2)
    ├── variant_a_minimal.txt               # Prompt Variant A (minimal)
    ├── variant_b_paraphrased.txt           # Prompt Variant B (paraphrased)
    └── variant_c_reordered.txt             # Prompt Variant C (reordered categories)
```

## Data

### Source Dataset

The original dataset of ~500,000 ChatGPT-related tweets (January–March 2023) is publicly available on Kaggle under a **CC0: Public Domain** license:

> Ansari, K. (2023). _500k ChatGPT-related Tweets Jan–Mar 2023_. Kaggle.  
> https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023

Download the source CSV and place it as `data/Twitter_Jan_Mar.csv` before running the pipeline.

### Provided Data Files

> **Note:** In compliance with the platform's data redistribution policies, the provided CSV files contain only tweet IDs and associated labels. Tweet text (`content`) and user information (`username`) have been removed. To reconstruct the full dataset, first download the source CSV from Kaggle (see above), then join on the `id` column to recover tweet content before running the pipeline.

| File                         | Description                                                                                       | Rows   |
| ---------------------------- | ------------------------------------------------------------------------------------------------- | ------ |
| `gold_validation_385.csv`    | Validation sample with human coder labels (SY, SH), Gold labels, GPT-4.1 and GPT-3.5-turbo labels | 385    |
| `gpt4_1_full_labels.csv`     | Full candidate corpus with GPT-4.1 classification labels                                          | 48,398 |
| `gpt3_5_full_labels.csv`     | Full candidate corpus with GPT-3.5-turbo classification labels                                    | 48,398 |
| `filtering_audit_100.csv`    | Random sample from excluded tweets, coded for risk presence                                       | 100    |
| `topic_category_mapping.csv` | Topic-to-category mapping results (two independent coders + Gold)                                 | 33     |

## Pipeline Overview

The pipeline consists of two stages:

### Stage 1: LLM-Based Risk Discourse Classification

1. **`01_keyword_prescreening.py`** — Applies keyword-based pre-screening to the raw ~500K tweets, producing a candidate corpus of 48,398 tweets.
2. **`02_llm_classification.py`** — Classifies the candidate corpus using GPT-4.1 and GPT-3.5-turbo via the OpenAI API (binary: 0 = non-risk, 1 = ethical risk discourse).
3. **`03_classification_evaluation.py`** — Evaluates model performance against the gold validation set (Accuracy, Precision, Recall, F1-score, Cohen's κ, confusion matrices).
4. **`04_mcnemar_test.py`** — McNemar's exact test comparing error distributions between GPT-4.1 and GPT-3.5-turbo.
5. **`05_prompt_robustness.py`** — Prompt robustness check (3 variants) and inter-run consistency check (3 repeated runs).

### Stage 2: BERTopic-Based Structural Analysis

6. **`06_bertopic_analysis.py`** — Text preprocessing, BERTopic modeling with sensitivity analysis across `min_cluster_size` values (50–100), coherence/diversity evaluation, and final topic extraction.

## Requirements

```
python >= 3.9
openai >= 1.0
pandas
numpy
scikit-learn
statsmodels
matplotlib
bertopic
sentence-transformers
hdbscan
umap-learn
gensim
spacy
```

Install dependencies:

```bash
pip install openai pandas numpy scikit-learn statsmodels matplotlib bertopic sentence-transformers hdbscan umap-learn gensim spacy
python -m spacy download en_core_web_sm
```

## Usage

### Setting up the OpenAI API key

Scripts that call the OpenAI API (`02`, `05`) require an API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Running the pipeline

```bash
# Stage 1
python code/01_keyword_prescreening.py
python code/02_llm_classification.py
python code/03_classification_evaluation.py
python code/04_mcnemar_test.py
python code/05_prompt_robustness.py

# Stage 2
python code/06_bertopic_analysis.py
```

> **Note:** Scripts `02` and `05` make API calls to OpenAI and may take several hours to complete due to rate limits. The pre-computed classification results are provided in `data/` for reproducibility without API access.

## Classification Prompt

The primary classification prompt used for both GPT-4.1 and GPT-3.5-turbo is provided in `prompts/primary_prompt.txt`. The full prompt text is also available in the paper (Table A2).

## Ethical Risk Categories

The five ethical risk dimensions used for classification and topic-to-category mapping:

1. **Technical safety** — hallucinations, harmful content generation, security vulnerabilities
2. **Privacy and data misuse** — personal data leakage, unauthorized data use
3. **Fairness and discrimination** — algorithmic bias, discriminatory outcomes, copyright violations
4. **Malicious misuse** — deepfakes, misinformation, jailbreak exploitation, criminal applications
5. **Societal and democratic risks** — labor displacement, threats to democracy, digital divide, human rights concerns

## Key Parameters

| Component          | Parameter                                      | Value                 |
| ------------------ | ---------------------------------------------- | --------------------- |
| LLM classification | Temperature                                    | 0.0                   |
| Sentence embedding | Model                                          | `all-mpnet-base-v2`   |
| UMAP               | n_neighbors / n_components / min_dist / metric | 15 / 5 / 0.0 / cosine |
| HDBSCAN            | min_cluster_size (final)                       | 90                    |
| c-TF-IDF           | BM25 weighting + reduce_frequent_words         | enabled               |
| CountVectorizer    | ngram_range                                    | (1, 2)                |

## Citation

```bibtex
@article{kim2026reproducible,
  title={A Reproducible and Human-Validated Pipeline for LLM-Based Classification of Ethical Risk Discourse in Large-Scale Social Media Data},
  author={Kim, Soyon and Kim, Soo Hyung and Lee, Bong Gyou},
  journal={PeerJ Computer Science},
  year={2026},
  note={Under review}
}
```

## License

The source tweet data is released under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).  
Code in this repository is released under the [MIT License](LICENSE).
