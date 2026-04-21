import re
import pandas as pd
from pathlib import Path

# Paths
INPUT_PATH = Path("../data/Twitter_Jan_Mar.csv")
OUTPUT_PATH = Path("../data/candidate_corpus.csv")

TEXT_COL = "content"

# Load raw data
df = pd.read_csv(INPUT_PATH)
print(f"[1] Loaded raw data: {len(df):,} rows")

# Basic cleaning
required_cols = ["date", "id", "content", "username"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df["username"] = df["username"].fillna("").astype(str)
print(f"[2] After basic cleaning: {len(df):,}")

# Remove URL-only tweets
url_only = re.compile(r"^\s*(https?://\S+|www\.\S+)\s*$", flags=re.IGNORECASE)
n_before = len(df)
df = df[~df[TEXT_COL].str.match(url_only, na=False)].copy()
print(f"[3] Removed URL-only tweets: {n_before - len(df):,} | Remaining: {len(df):,}")

# Remove duplicates
n_before = len(df)
df = df.drop_duplicates(subset=[TEXT_COL], keep="first").copy()
print(f"[4] Removed duplicates: {n_before - len(df):,} | Remaining: {len(df):,}")

# Remove short tweets (<5 words)
n_before = len(df)
df["word_count"] = df[TEXT_COL].str.split().str.len()
df = df[df["word_count"] >= 5].drop(columns=["word_count"]).copy()
print(f"[5] Removed <5-word tweets: {n_before - len(df):,} | Remaining: {len(df):,}")

# Remove emojis
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
df[TEXT_COL] = df[TEXT_COL].apply(lambda t: emoji_pattern.sub("", t))
print(f"[6] After emoji removal: {len(df):,}")

# Keyword-based pre-screening
KEYWORD_STEMS = [
    "hallucinat", "misinform", "disinform", "bias", "discrimin",
    "stereotyp", "inequit", "copyright", "infring",
    "manipulat", "propagand", "privac", "surveil",
    "automat", "labor", "opac", "transparen", "interpret", "explain",
    "account", "trust", "jailbreak", "guardrail", "inject",
    "secur", "saf", "sustain", "ethic", "govern", "regulat",
    "risk", "fair", "harm", "unsaf", "respons", "autonom", "dignit", "justic",
]

pattern = r"\b(?:%s)\w*\b" % "|".join(map(re.escape, KEYWORD_STEMS))
regex = re.compile(pattern, flags=re.IGNORECASE)

df[TEXT_COL] = df[TEXT_COL].str.replace(r"[-–—]", " ", regex=True)
mask = df[TEXT_COL].str.contains(regex, na=False)
filtered = df[mask].copy()
filtered["matched_keywords"] = (
    filtered[TEXT_COL]
    .str.findall(regex)
    .apply(lambda lst: [s.lower() for s in lst])
)

print(f"[7] After keyword filtering: {len(filtered):,} ({len(filtered)/len(df)*100:.1f}%)")

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
filtered.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved candidate corpus to {OUTPUT_PATH}")
