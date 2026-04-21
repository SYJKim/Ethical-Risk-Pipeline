import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score,
)

HUMAN_ANNOTATION = Path("../data/human_annotation_385.csv")
OUT_PERF = Path("../data/supervised_baseline_performance.csv")
OUT_OOF  = Path("../data/supervised_baseline_oof_predictions.csv")

N_FOLDS = 5
SEED = 42

BERTWEET_MODEL = "vinai/bertweet-base"
ROBERTA_MODEL  = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_predictions(y_true, y_pred, model_name):
    return {
        "Model":       model_name,
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "Cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def train_transformer_fold(model_name, train_texts, train_labels, test_texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)

    train_ds = TweetDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            optimizer.step()

    test_ds = TweetDataset(test_texts, np.zeros(len(test_texts), dtype=int), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return np.array(preds)


if __name__ == "__main__":
    df = pd.read_csv(HUMAN_ANNOTATION)
    if "content" not in df.columns:
        raise RuntimeError(
            "human_annotation_385.csv must contain a 'content' column. "
            "Rehydrate tweet text from the Kaggle source (Ansari, 2023) first — "
            "see README section 'Data rehydration'."
        )

    texts       = df["content"].values
    labels      = df["Gold_label"].values
    gpt4_preds  = df["gpt_4_1_label"].values
    gpt35_preds = df["gpt_3_5_turbo_label"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(texts, labels))

    print("=" * 60)
    print("TF-IDF + Logistic Regression (stratified 5-fold CV)")
    print("=" * 60)
    tfidf_oof = np.zeros(len(labels), dtype=int)
    for fold_i, (tr, te) in enumerate(folds):
        print(f"  Fold {fold_i + 1}/{N_FOLDS}")
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_tr = vec.fit_transform(texts[tr])
        X_te = vec.transform(texts[te])
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_tr, labels[tr])
        tfidf_oof[te] = clf.predict(X_te)

    print("\n" + "=" * 60)
    print("BERTweet fine-tuning (stratified 5-fold CV)")
    print("=" * 60)
    bertweet_oof = np.zeros(len(labels), dtype=int)
    for fold_i, (tr, te) in enumerate(folds):
        print(f"  Fold {fold_i + 1}/{N_FOLDS}")
        bertweet_oof[te] = train_transformer_fold(
            BERTWEET_MODEL, texts[tr], labels[tr], texts[te]
        )

    print("\n" + "=" * 60)
    print("RoBERTa fine-tuning (stratified 5-fold CV)")
    print("=" * 60)
    roberta_oof = np.zeros(len(labels), dtype=int)
    for fold_i, (tr, te) in enumerate(folds):
        print(f"  Fold {fold_i + 1}/{N_FOLDS}")
        roberta_oof[te] = train_transformer_fold(
            ROBERTA_MODEL, texts[tr], labels[tr], texts[te]
        )

    print("\n" + "=" * 60)
    print("Overall performance (out-of-fold, n = 385)")
    print("=" * 60)
    results = [
        evaluate_predictions(labels, tfidf_oof,     "TF-IDF + LR"),
        evaluate_predictions(labels, bertweet_oof,  "BERTweet"),
        evaluate_predictions(labels, roberta_oof,   "RoBERTa"),
        evaluate_predictions(labels, gpt35_preds,   "GPT-3.5-turbo (zero-shot)"),
        evaluate_predictions(labels, gpt4_preds,    "GPT-4.1 (zero-shot)"),
    ]
    perf_df = pd.DataFrame(results)
    print(perf_df.round(4).to_string(index=False))
    perf_df.to_csv(OUT_PERF, index=False)
    print(f"\nSaved: {OUT_PERF}")

    oof_df = df[["date", "id", "Gold_label"]].copy()
    oof_df["tfidf_lr_pred"] = tfidf_oof
    oof_df["bertweet_pred"] = bertweet_oof
    oof_df["roberta_pred"]  = roberta_oof
    oof_df["gpt35_pred"]    = gpt35_preds
    oof_df["gpt4_pred"]     = gpt4_preds
    oof_df.to_csv(OUT_OOF, index=False)
    print(f"Saved: {OUT_OOF}")
