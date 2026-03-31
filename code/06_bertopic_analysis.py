"""
06_bertopic_analysis.py
Stage 2: BERTopic-based structural analysis of ethical risk discourse.

Input:  data/gpt4_1_full_labels.csv
Output: results/sensitivity_metrics.csv
        results/final_mcs_90/ (topic_info, topic_keywords, document_info)
        results/sensitivity_summary.png
"""

import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP
import spacy

# ── Configuration ──────────────────────────────────────────
INPUT_CSV = Path("../data/gpt4_1_full_labels.csv")
TEXT_COLUMN = "content"
LABEL_COLUMN = "gpt_label"
POSITIVE_LABEL = 1

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = OUTPUT_DIR / "diagnostic_by_mcs"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# Set to None for sensitivity analysis only; set to 90 for final model
FINAL_MIN_CLUSTER_SIZE = 90
CLUSTER_SIZES = [50, 60, 70, 80, 90, 100]

RANDOM_STATE = 42
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

# HDBSCAN parameters
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# Vectorizer parameters
NGRAM_RANGE = (1, 2)
TOP_N_WORDS = 10

# Domain-specific stopwords (Table A3)
CUSTOM_STOPWORDS = [
    "chatgpt", "ai", "gpt", "chat", "openai", "google", "microsoft", "bing", "bard", "midjourney",
    "twitter", "blockchain", "bitcoin", "btc", "eth", "nft", "ml", "seo", "amp", "gt",
    "openaichatgpt", "app", "bot", "good", "way", "better", "interesting",
    "thank", "thanks", "amazing", "fun", "lol", "maybe", "thoughts", "thinking",
    "pretty", "wrong", "cool", "bad", "feel", "important", "short", "big",
    "sure", "high", "just", "called", "end", "far", "stuff", "looks", "thought",
    "today", "going", "ve", "let", "want", "day", "got", "try", "ll", "thing",
    "years", "week", "lot", "soon", "coming", "come", "ways", "year", "getting",
    "tried", "vs", "tweet", "days", "trying", "having", "makes", "isn", "won",
    "needs", "months", "yes", "taking", "month", "little", "instead", "hours",
    "mean", "billion", "add", "hey", "million", "pm", "gonna", "comes", "probably",
    "set", "s", "br", "wa", "don", "ha", "doe", "m", "didn", "doesn", "wan", "na",
]


# ── Utilities ──────────────────────────────────────────────
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def build_stopwords(nlp):
    return set(ENGLISH_STOP_WORDS) | set(nlp.Defaults.stop_words) | {w.lower() for w in CUSTOM_STOPWORDS}


def preprocess(df, text_col, stopwords, nlp):
    """Clean, deduplicate, lemmatize, and remove stopwords."""
    mention_hashtag = re.compile(r"(@\w+)|(#[\w_]+)")
    url_pattern = re.compile(r"https?://\S+|www\.\S+")

    work = df.copy()
    work["cleaned_content"] = (
        work[text_col].fillna("").astype(str)
        .str.replace(url_pattern, " ", regex=True)
        .str.replace(mention_hashtag, " ", regex=True)
        .str.replace(r"[^A-Za-z\s]", " ", regex=True)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    work = work.drop_duplicates(subset=["cleaned_content"]).reset_index(drop=True)

    def lemmatize(text):
        doc = nlp(text)
        return [
            t.lemma_.lower().strip() for t in doc
            if t.lemma_.lower().strip()
            and t.lemma_.isalpha()
            and not t.is_space and not t.is_punct
            and not t.like_url and not t.like_email
            and t.lemma_.lower().strip() not in stopwords
        ]

    work["tokens"] = work["cleaned_content"].apply(lemmatize)
    work["finalized_content"] = work["tokens"].apply(lambda x: " ".join(x))
    work = work[work["finalized_content"].str.strip().ne("")].reset_index(drop=True)
    return work


def fit_bertopic(docs, embeddings, embedding_model, vectorizer, ctfidf, min_cluster_size, top_n):
    """Fit a BERTopic model with specified min_cluster_size."""
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC, cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    model = BERTopic(
        embedding_model=embedding_model, umap_model=umap_model,
        hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
        ctfidf_model=ctfidf, top_n_words=top_n, verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    return model, topics


def compute_coherence(model, tokenized_docs, dictionary, docs, top_n):
    """Compute C_v and C_NPMI coherence scores."""
    topic_words = []
    for tid in sorted(set(model.get_topics().keys()) - {-1}):
        words = [w for w, _ in model.get_topic(tid)[:top_n]]
        topic_words.append(words)
    if not topic_words:
        return 0.0, 0.0

    cv = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence="c_v").get_coherence()
    cnpmi = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence="c_npmi").get_coherence()
    return cv, cnpmi


def compute_diversity(model, top_n):
    """Topic diversity: proportion of unique words across all topics."""
    all_words, unique_words = [], set()
    for tid in sorted(set(model.get_topics().keys()) - {-1}):
        words = [w for w, _ in model.get_topic(tid)[:top_n]]
        all_words.extend(words)
        unique_words.update(words)
    return len(unique_words) / len(all_words) if all_words else 0.0


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load and filter
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded: {len(df):,} rows")
    if LABEL_COLUMN in df.columns:
        df = df[df[LABEL_COLUMN] == POSITIVE_LABEL].reset_index(drop=True)
        print(f"After filtering gpt_label == {POSITIVE_LABEL}: {len(df):,}")

    # 2. Preprocess
    nlp = load_spacy_model()
    all_stopwords = build_stopwords(nlp)
    df_model = preprocess(df, TEXT_COLUMN, all_stopwords, nlp)
    docs = df_model["finalized_content"].tolist()
    print(f"After preprocessing: {len(docs):,} documents")

    # 3. Tokenize for coherence
    analyzer = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words=list(all_stopwords)).build_analyzer()
    tokenized_docs = [analyzer(doc) for doc in docs]
    dictionary = Dictionary(tokenized_docs)

    # 4. Embed
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words=list(all_stopwords))
    ctfidf = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

    # 5. Sensitivity analysis
    print("\n=== Sensitivity Analysis (Table A6) ===")
    rows = []
    for size in CLUSTER_SIZES:
        print(f"  min_cluster_size = {size}...")
        model, topics = fit_bertopic(docs, embeddings, embedding_model, vectorizer, ctfidf, size, TOP_N_WORDS)
        n_topics = len(set(topics) - {-1})
        outlier_ratio = sum(1 for t in topics if t == -1) / len(topics)
        cv, cnpmi = compute_coherence(model, tokenized_docs, dictionary, docs, TOP_N_WORDS)
        td = compute_diversity(model, TOP_N_WORDS)

        row = {"min_cluster_size": size, "n_topics": n_topics, "outlier_ratio": round(outlier_ratio, 4),
               "c_v": round(cv, 4), "c_npmi": round(cnpmi, 4), "topic_diversity": round(td, 4)}
        rows.append(row)

        # Save diagnostic outputs
        diag_path = DIAG_DIR / f"mcs_{size}"
        diag_path.mkdir(parents=True, exist_ok=True)
        model.get_topic_info().to_csv(diag_path / "topic_info.csv", index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "sensitivity_metrics.csv", index=False)
    print(results_df.to_string(index=False))

    # 6. Sensitivity figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, ["n_topics", "outlier_ratio", "c_v", "topic_diversity"]):
        ax.plot(results_df["min_cluster_size"], results_df[metric], marker="o")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / "sensitivity_summary.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'sensitivity_summary.png'}")

    # 7. Final model
    if FINAL_MIN_CLUSTER_SIZE is not None:
        print(f"\n=== Final Model (min_cluster_size = {FINAL_MIN_CLUSTER_SIZE}) ===")
        final_model, final_topics = fit_bertopic(
            docs, embeddings, embedding_model, vectorizer, ctfidf, FINAL_MIN_CLUSTER_SIZE, TOP_N_WORDS
        )
        final_dir = OUTPUT_DIR / f"final_mcs_{FINAL_MIN_CLUSTER_SIZE}"
        final_dir.mkdir(parents=True, exist_ok=True)

        topic_info = final_model.get_topic_info()
        topic_info.to_csv(final_dir / "topic_info.csv", index=False)

        # Topic keywords
        kw_rows = []
        for tid in sorted(set(final_topics) - {-1}):
            words = [w for w, _ in final_model.get_topic(tid)[:TOP_N_WORDS]]
            kw_rows.append({"topic": tid, "top_words": " | ".join(words)})
        pd.DataFrame(kw_rows).to_csv(final_dir / "topic_keywords.csv", index=False)

        # Document-topic assignments
        doc_df = df_model.copy()
        doc_df["topic"] = final_topics
        doc_df.to_csv(final_dir / "document_info.csv", index=False)

        n_topics = len(set(final_topics) - {-1})
        outlier_n = sum(1 for t in final_topics if t == -1)
        print(f"  Topics: {n_topics}")
        print(f"  Outliers: {outlier_n} ({outlier_n/len(final_topics)*100:.1f}%)")
        print(f"  Non-outlier documents: {len(final_topics) - outlier_n}")
        print(f"  Saved to: {final_dir.resolve()}")
