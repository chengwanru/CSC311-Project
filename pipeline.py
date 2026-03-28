"""
pipeline.py
===========
Single canonical module for all data loading, cleaning, splitting, and
feature engineering used across the CSC311 project.

Replaces the former trio of:
  preprocessing.py   — column constants, clean(), fit_preprocess()
  data_splitting.py  — regular_split(), grouped_kfold_split()
  run_preprocess.py  — one-shot preprocessing utility

Training scripts (naive_bayes.py, stacking_ensemble.py, stacking_experiments.py)
import from this module for shared preprocessing and splits.

pred.py is intentionally self-contained (no project imports allowed).

Sections
--------
1. Constants & column names
2. Row-level cleaning  (clean / clean_preprocessed)
3. Data splitting      (regular_split / grouped_kfold_split)
4. Feature helpers     (extract_numeric, extract_likert, …)
5. TF-IDF helpers      (normalize_text, build_vocab_idf, tfidf_matrix)
6. Training pipeline   (fit_state, transform_df)
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

# ============================================================
# 1.  CONSTANTS & COLUMN NAMES
# ============================================================

# --- File paths & global seeds ---
CSV_PATH     = "training_data.csv"   # raw survey data (all models)
RANDOM_STATE = 42

# --- Column names in training_data.csv (full survey question text) ---
COL_ID         = "unique_id"
COL_TARGET     = "Painting"
COL_EMOTION    = "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?"
COL_DESC       = "Describe how this painting makes you feel."
COL_SOMBRE     = "This art piece makes me feel sombre."
COL_CONTENT    = "This art piece makes me feel content."
COL_CALM       = "This art piece makes me feel calm."
COL_UNEASY     = "This art piece makes me feel uneasy."
COL_N_COLOURS  = "How many prominent colours do you notice in this painting?"
COL_N_OBJECTS  = "How many objects caught your eye in the painting?"
COL_PRICE      = "How much (in Canadian dollars) would you be willing to pay for this painting?"
COL_ROOM       = "If you could purchase this painting, which room would you put that painting in?"
COL_WHO        = "If you could view this art in person, who would you want to view it with?"
COL_SEASON     = "What season does this art piece remind you of?"
COL_FOOD       = "If this painting was a food, what would be?"
COL_SOUNDTRACK = (
    "Imagine a soundtrack for this painting. "
    "Describe that soundtrack without naming any objects in the painting."
)

NUMERIC_COLS = [COL_EMOTION, COL_N_COLOURS, COL_N_OBJECTS]
LIKERT_COLS  = [COL_SOMBRE, COL_CONTENT, COL_CALM, COL_UNEASY]
MULTI_COLS   = [COL_ROOM, COL_WHO, COL_SEASON]
TEXT_COLS    = [COL_DESC, COL_FOOD, COL_SOUNDTRACK]

# Required to be non-null for a row to survive clean()
REQUIRED_FOR_CLEAN = (
    [COL_ID, COL_TARGET, COL_EMOTION, COL_N_COLOURS, COL_N_OBJECTS] + LIKERT_COLS
)


# ============================================================
# 2.  ROW-LEVEL CLEANING
# ============================================================

def clean(df):
    """
    Drop rows from training_data.csv that are missing any required numeric or
    Likert column; fill text / multi-label columns with empty strings.

    Use the returned DataFrame as input to regular_split or grouped_kfold_split.
    """
    df_clean = df.dropna(subset=REQUIRED_FOR_CLEAN).copy()
    for c in TEXT_COLS:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna("")
    for c in MULTI_COLS:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna("")
    return df_clean



# ============================================================
# 3.  DATA SPLITTING
# ============================================================

def regular_split(df, id_col="unique_id", random_state=42):
    """
    60 / 20 / 20 person-level split.

    Shuffles unique person IDs with the given seed, then assigns
    60 % to train, 20 % to val, and the remaining 20 % to test.
    All rows belonging to a person land in the same split.

    Returns (train_df, val_df, test_df).
    """
    df_clean = df.dropna(subset=[id_col]).copy()
    rng = np.random.RandomState(random_state)

    unique_ids = df_clean[id_col].unique()
    rng.shuffle(unique_ids)

    n_ids   = len(unique_ids)
    n_train = int(n_ids * 0.60)
    n_val   = int(n_ids * 0.20)

    train_ids = unique_ids[:n_train]
    val_ids   = unique_ids[n_train:n_train + n_val]
    test_ids  = unique_ids[n_train + n_val:]

    return (
        df_clean[df_clean[id_col].isin(train_ids)].copy(),
        df_clean[df_clean[id_col].isin(val_ids)].copy(),
        df_clean[df_clean[id_col].isin(test_ids)].copy(),
    )


def grouped_kfold_split(df, id_col="unique_id", k=5, random_state=42):
    """
    Person-level k-fold CV with a fixed 20 % held-out test set.

    The 20 % test IDs are drawn first (same for every fold).
    The remaining 80 % are split into k equal folds; each iteration
    yields one fold as validation and the remaining k-1 as training.

    Yields k tuples of (train_df, val_df, test_df).
    """
    rng = np.random.RandomState(random_state)

    unique_ids = df[id_col].dropna().unique()
    rng.shuffle(unique_ids)

    n_ids          = len(unique_ids)
    n_trainval_ids = int(n_ids * 0.80)

    trainval_ids = unique_ids[:n_trainval_ids]
    test_ids     = unique_ids[n_trainval_ids:]

    test_df = df[df[id_col].isin(test_ids)].copy()
    folds   = np.array_split(trainval_ids, k)

    for i in range(k):
        val_ids   = folds[i]
        train_ids = np.concatenate([folds[j] for j in range(k) if j != i])
        yield (
            df[df[id_col].isin(train_ids)].copy(),
            df[df[id_col].isin(val_ids)].copy(),
            test_df,
        )


# ============================================================
# 4.  FEATURE HELPERS
# ============================================================

def extract_numeric(series, clip_max=None, impute=None):
    """Parse the first number from each cell; clip and impute as specified."""
    fill = float(impute) if impute is not None else 0.0
    out = []
    for v in series:
        if pd.isna(v):
            out.append(fill)
            continue
        m = re.search(r"\d+(?:,\d{3})*\.?\d*", str(v))
        if m:
            x = float(m.group().replace(",", ""))
            if clip_max is not None and x > clip_max:
                x = clip_max
            out.append(x)
        else:
            out.append(fill)
    return np.array(out, dtype=float)


def extract_likert(series, impute=None):
    """Extract a leading 1–5 digit from each Likert response cell."""
    fill = int(round(np.clip(float(impute), 1, 5))) if impute is not None else 0
    out = []
    for v in series:
        if pd.isna(v):
            out.append(fill)
            continue
        m = re.search(r"^([1-5])", str(v).strip())
        out.append(int(m.group(1)) if m else fill)
    return np.array(out, dtype=float)


def get_categories(series):
    """Collect all unique comma-separated tokens seen in a column."""
    cats = set()
    for v in series.dropna():
        for part in str(v).split(","):
            part = part.strip()
            if part:
                cats.add(part)
    return sorted(cats)


def multi_hot(series, categories):
    """Binary indicator matrix for comma-separated categorical column."""
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((len(series), len(categories)), dtype=float)
    for i, v in enumerate(series):
        if pd.isna(v):
            continue
        for part in str(v).split(","):
            key = part.strip()
            if key in cat_to_idx:
                mat[i, cat_to_idx[key]] = 1.0
    return mat


# ============================================================
# 5.  TF-IDF HELPERS
# ============================================================

def normalize_text(x):
    """Lowercase, strip non-alphanumeric, collapse whitespace."""
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text):
    """Word-boundary tokenizer (mirrors sklearn token_pattern)."""
    return re.findall(r"(?u)\b\w+\b", str(text).lower())


def build_vocab_idf(texts, max_features, min_df):
    """
    Build vocabulary and smooth IDF weights from a list of raw strings.

    Uses sklearn-style smooth IDF: log((1 + n) / (1 + df)) + 1.
    Returns (vocab, idf) where vocab is a list of strings and
    idf is a float64 array of the same length.
    """
    docs = [tokenize(t) for t in texts]
    n_docs = len(docs)

    df_count = Counter()
    for toks in docs:
        df_count.update(set(toks))

    kept = [w for w, c in df_count.items() if c >= min_df]
    kept.sort(key=lambda w: (-df_count[w], w))
    if max_features is not None:
        kept = kept[:max_features]

    idf = np.array(
        [np.log((1.0 + n_docs) / (1.0 + df_count[w])) + 1.0 for w in kept],
        dtype=float,
    )
    return kept, idf


def tfidf_matrix(texts, vocab, idf):
    """
    TF-IDF feature matrix with L2-normalised rows.

    texts : list of strings (already normalised with normalize_text)
    vocab : list of token strings
    idf   : float array matching vocab
    """
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, t in enumerate(texts):
        counts = Counter(tokenize(t))
        for tok, cnt in counts.items():
            j = vocab_to_idx.get(tok)
            if j is not None:
                X[i, j] = float(cnt) * float(idf[j])
        nrm = np.linalg.norm(X[i])
        if nrm > 0:
            X[i] /= nrm
    return X


# ============================================================
# 6.  TRAINING PIPELINE  (fit on train only)
# ============================================================

def fit_state(train_df, max_features=6000, min_df=1):
    """
    Fit all preprocessing statistics on train_df only.

    Returns a state dict that can be passed to transform_df.
    Statistics computed:
      - class label → integer index mapping
      - per-numeric-column: median imputation, 97th-percentile clip, mean, std
      - per-Likert-column: median imputation
      - price column: same as numeric
      - multi-hot categories for room / who / season
      - vocabulary and IDF weights for TF-IDF
    """
    state = {}
    state["classes"]      = sorted(train_df[COL_TARGET].unique())
    state["class_to_idx"] = {c: i for i, c in enumerate(state["classes"])}

    # Numeric columns
    num_medians, num_clips, num_means, num_stds = [], [], [], []
    for c in NUMERIC_COLS:
        non_null = train_df[c].dropna()
        med  = float(np.median(extract_numeric(non_null))) if len(non_null) else 0.0
        raw  = extract_numeric(train_df[c], impute=med)
        clip = np.percentile(raw[raw > 0] if np.any(raw > 0) else raw, 97)
        clip = 1.0 if clip == 0 else float(clip)
        x    = extract_numeric(train_df[c], clip_max=clip, impute=med)
        num_medians.append(med)
        num_clips.append(clip)
        num_means.append(float(np.mean(x)))
        num_stds.append(float(np.std(x)) or 1.0)
    state["num_medians"] = np.array(num_medians, dtype=float)
    state["num_clips"]   = np.array(num_clips,   dtype=float)
    state["num_means"]   = np.array(num_means,   dtype=float)
    state["num_stds"]    = np.array(num_stds,    dtype=float)

    # Likert columns
    likert_medians = []
    for c in LIKERT_COLS:
        non_null = train_df[c].dropna()
        med = float(np.median(extract_likert(non_null))) if len(non_null) else 3.0
        likert_medians.append(med)
    state["likert_medians"] = np.array(likert_medians, dtype=float)

    # Price
    pnn          = train_df[COL_PRICE].dropna()
    price_median = float(np.median(extract_numeric(pnn))) if len(pnn) else 0.0
    price_raw    = extract_numeric(train_df[COL_PRICE], impute=price_median)
    price_clip   = np.percentile(
        price_raw[price_raw > 0] if np.any(price_raw > 0) else price_raw, 97
    )
    price_clip = 1.0 if price_clip == 0 else float(price_clip)
    price_x    = extract_numeric(train_df[COL_PRICE], clip_max=price_clip,
                                 impute=price_median)
    state["price_median"] = float(price_median)
    state["price_clip"]   = float(price_clip)
    state["price_mean"]   = float(np.mean(price_x))
    state["price_std"]    = float(np.std(price_x)) or 1.0

    # Multi-hot categories
    state["room_cats"]   = get_categories(train_df[COL_ROOM])
    state["who_cats"]    = get_categories(train_df[COL_WHO])
    state["season_cats"] = get_categories(train_df[COL_SEASON])

    # TF-IDF
    train_text = (
        train_df[COL_DESC].map(normalize_text).astype(str) + " "
        + train_df[COL_FOOD].map(normalize_text).astype(str) + " "
        + train_df[COL_SOUNDTRACK].map(normalize_text).astype(str)
    )
    vocab, idf = build_vocab_idf(train_text.values,
                                  max_features=max_features, min_df=min_df)
    state["vocab"]        = vocab
    state["idf"]          = idf
    state["max_features"] = max_features
    state["min_df"]       = min_df

    return state


def transform_df(df, state):
    """
    Apply a fitted state to a DataFrame, returning (X, y).

    All statistics (clips, means, stds, vocab, …) come from the state
    fitted on the training split — no data leakage.
    y is the integer label array (class_to_idx applied to COL_TARGET).
    """
    x_num = np.column_stack([
        (
            extract_numeric(df[c],
                            clip_max=float(state["num_clips"][j]),
                            impute=float(state["num_medians"][j]))
            - state["num_means"][j]
        ) / state["num_stds"][j]
        for j, c in enumerate(NUMERIC_COLS)
    ])
    x_likert = np.column_stack([
        extract_likert(df[c], impute=float(state["likert_medians"][j]))
        for j, c in enumerate(LIKERT_COLS)
    ])
    pr = extract_numeric(df[COL_PRICE],
                         clip_max=state["price_clip"],
                         impute=state["price_median"])
    x_price  = ((pr - state["price_mean"]) / state["price_std"]).reshape(-1, 1)

    x_room   = multi_hot(df[COL_ROOM],   state["room_cats"])
    x_who    = multi_hot(df[COL_WHO],    state["who_cats"])
    x_season = multi_hot(df[COL_SEASON], state["season_cats"])

    text = (
        df[COL_DESC].map(normalize_text).astype(str) + " "
        + df[COL_FOOD].map(normalize_text).astype(str) + " "
        + df[COL_SOUNDTRACK].map(normalize_text).astype(str)
    )
    x_tfidf = tfidf_matrix(text.values, state["vocab"], state["idf"])

    X = np.hstack([x_num, x_likert, x_price, x_room, x_who, x_season, x_tfidf])
    y = np.array([state["class_to_idx"][c] for c in df[COL_TARGET]])
    return X, y
