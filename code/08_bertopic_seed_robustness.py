import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.corpora import Dictionary
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_main = import_module("07_bertopic_analysis")

INPUT_CSV = Path("../data/gpt_4_1_full_labels.csv")
TEXT_COLUMN  = "content"
LABEL_COLUMN = "gpt_4_1_label"
POSITIVE_LABEL = 1

OUTPUT_DIR = Path("../results/seed_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_MIN_CLUSTER_SIZE = 90
PRIMARY_SEED = 42
N_SEEDS = 5

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
UMAP_N_NEIGHBORS  = _main.UMAP_N_NEIGHBORS
UMAP_N_COMPONENTS = _main.UMAP_N_COMPONENTS
UMAP_MIN_DIST     = _main.UMAP_MIN_DIST
UMAP_METRIC       = _main.UMAP_METRIC

HDBSCAN_MIN_SAMPLES = _main.HDBSCAN_MIN_SAMPLES
HDBSCAN_METRIC      = _main.HDBSCAN_METRIC
HDBSCAN_CLUSTER_SELECTION_METHOD = _main.HDBSCAN_CLUSTER_SELECTION_METHOD

NGRAM_RANGE = _main.NGRAM_RANGE
TOP_N_WORDS = _main.TOP_N_WORDS


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    if TEXT_COLUMN not in df.columns:
        raise RuntimeError(
            f"{INPUT_CSV} must contain a '{TEXT_COLUMN}' column. "
            "Rehydrate tweet text from the Kaggle source (Ansari, 2023) first — "
            "see README section 'Data rehydration'."
        )
    print(f"Loaded: {len(df):,} rows")
    df = df[df[LABEL_COLUMN] == POSITIVE_LABEL].reset_index(drop=True)
    print(f"After filtering {LABEL_COLUMN} == {POSITIVE_LABEL}: {len(df):,}")

    nlp = _main.load_spacy_model()
    all_stopwords = _main.build_stopwords(nlp)
    df_model = _main.preprocess(df, TEXT_COLUMN, all_stopwords, nlp)
    docs = df_model["finalized_content"].tolist()
    print(f"After preprocessing: {len(docs):,} documents")

    analyzer = CountVectorizer(
        ngram_range=NGRAM_RANGE, stop_words=list(all_stopwords),
    ).build_analyzer()
    tokenized_docs = [analyzer(doc) for doc in docs]
    dictionary = Dictionary(tokenized_docs)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(
        docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
    )
    vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words=list(all_stopwords))
    ctfidf = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

    pool = [s for s in range(1, 10_000) if s != PRIMARY_SEED]
    seeds = random.sample(pool, N_SEEDS)
    print(f"\nUMAP seeds to evaluate (excluding primary seed {PRIMARY_SEED}): {seeds}")

    rows = []
    for seed in seeds:
        print(f"\n  seed = {seed}...")

        umap_model = UMAP(
            n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
            min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=seed,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=FIXED_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES,
            metric=HDBSCAN_METRIC, cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        )
        model = BERTopic(
            embedding_model=embedding_model, umap_model=umap_model,
            hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
            ctfidf_model=ctfidf, top_n_words=TOP_N_WORDS, verbose=False,
        )
        topics, _ = model.fit_transform(docs, embeddings=embeddings)

        n_topics = len(set(topics) - {-1})
        outlier_ratio = sum(1 for t in topics if t == -1) / len(topics)
        cv, cnpmi = _main.compute_coherence(model, tokenized_docs, dictionary, TOP_N_WORDS)
        td = _main.compute_diversity(model, TOP_N_WORDS)

        rows.append({
            "seed":            seed,
            "n_topics":        n_topics,
            "outlier_ratio":   round(outlier_ratio, 4),
            "c_v":             round(cv, 4),
            "c_npmi":          round(cnpmi, 4),
            "topic_diversity": round(td, 4),
        })
        print(f"    n_topics={n_topics}, outlier={outlier_ratio:.3f}, "
              f"Cv={cv:.4f}, CNPMI={cnpmi:.4f}, TD={td:.4f}")

        topic_info = model.get_topic_info()
        kw_rows = []
        for tid in sorted(set(topics) - {-1}):
            words = [w for w, _ in model.get_topic(tid)[:TOP_N_WORDS]]
            count = int(topic_info.loc[topic_info["Topic"] == tid, "Count"].iloc[0])
            kw_rows.append({
                "Topic":    tid,
                "Count":    count,
                "TopWords": " | ".join(words),
            })
        pd.DataFrame(kw_rows).to_csv(
            OUTPUT_DIR / f"topic_keywords_seed_{seed}.csv", index=False,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "seed_robustness_summary.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Seed robustness summary (min_cluster_size = {FIXED_MIN_CLUSTER_SIZE})")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\n  n_topics        : {summary['n_topics'].mean():.2f} +/- {summary['n_topics'].std(ddof=1):.2f}")
    print(f"  outlier_ratio   : {summary['outlier_ratio'].mean():.4f} +/- {summary['outlier_ratio'].std(ddof=1):.4f}")
    print(f"  C_v             : {summary['c_v'].mean():.4f} +/- {summary['c_v'].std(ddof=1):.4f}")
    print(f"  C_NPMI          : {summary['c_npmi'].mean():.4f} +/- {summary['c_npmi'].std(ddof=1):.4f}")
    print(f"  topic_diversity : {summary['topic_diversity'].mean():.4f} +/- {summary['topic_diversity'].std(ddof=1):.4f}")
    print(f"\nSaved summary to {OUTPUT_DIR / 'seed_robustness_summary.csv'}")
