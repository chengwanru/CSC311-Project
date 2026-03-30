"""
Merged preprocessing + feature engineering pipeline.
Safe merge of your original scripts.

- Keeps your working pipeline intact
- Adds per-painting imputation (stronger)
- Keeps TF-IDF + multi-hot + scaling
"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# COLUMN DEFINITIONS (UNCHANGED)
# ============================================================

COL_ID = "unique_id"
COL_TARGET = "Painting"
COL_EMOTION = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
COL_DESC = "Describe how this painting makes you feel."
COL_SOMBRE = "This art piece makes me feel sombre."
COL_CONTENT = "This art piece makes me feel content."
COL_CALM = "This art piece makes me feel calm."
COL_UNEASY = "This art piece makes me feel uneasy."
COL_N_COLOURS = "How many prominent colours do you notice in this painting?"
COL_N_OBJECTS = "How many objects caught your eye in the painting?"
COL_PRICE = "How much (in Canadian dollars) would you be willing to pay for this painting?"
COL_ROOM = "If you could purchase this painting, which room would you put that painting in?"
COL_WHO = "If you could view this art in person, who would you want to view it with?"
COL_SEASON = "What season does this art piece remind you of?"
COL_FOOD = "If this painting was a food, what would be?"
COL_SOUNDTRACK = "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."

NUMERIC_COLS = [COL_EMOTION, COL_N_COLOURS, COL_N_OBJECTS]
LIKERT_COLS = [COL_SOMBRE, COL_CONTENT, COL_CALM, COL_UNEASY]
MULTI_COLS = [COL_ROOM, COL_WHO, COL_SEASON]
TEXT_COLS = [COL_DESC, COL_FOOD, COL_SOUNDTRACK]

# ============================================================
# PARSING FUNCTIONS (KEEPED + FIXED)
# ============================================================

def _extract_numeric(series):
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        m = re.search(r"\d+(?:,\d{3})*\.?\d*", str(v))
        if m:
            out.append(float(m.group().replace(",", "")))
        else:
            out.append(np.nan)
    return pd.Series(out, index=series.index)


def _extract_likert(series):
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        m = re.search(r"^([1-5])", str(v).strip())
        out.append(float(m.group(1)) if m else np.nan)
    return pd.Series(out, index=series.index)


def _extract_price(series, cap=10000):
    vals = _extract_numeric(series)
    vals = vals.clip(upper=cap)
    return np.log1p(vals)

# ============================================================
# CLEAN (SAFE)
# ============================================================

def clean(df):
    df = df.copy()

    for c in TEXT_COLS:
        df[c] = df[c].fillna("")

    for c in MULTI_COLS:
        df[c] = df[c].fillna("")

    return df

# ============================================================
# FIT (TRAIN ONLY)
# ============================================================

def fit_preprocess(train_df, max_features=6000, min_df=2):
    state = {}

    # Classes
    classes = sorted(train_df[COL_TARGET].unique())
    state["class_to_idx"] = {c: i for i, c in enumerate(classes)}

    # ---------- PER-PAINTING MEDIANS ----------
    paintings = train_df[COL_TARGET].unique()

    state["num_medians"] = {}
    for col in NUMERIC_COLS:
        parsed = _extract_numeric(train_df[col])
        state["num_medians"][col] = {
            p: parsed[train_df[COL_TARGET] == p].median()
            for p in paintings
        }

    state["likert_medians"] = {}
    for col in LIKERT_COLS:
        parsed = _extract_likert(train_df[col])
        state["likert_medians"][col] = {
            p: parsed[train_df[COL_TARGET] == p].median()
            for p in paintings
        }

    # ---------- CLIPPING + SCALING ----------
    num_data = []
    for col in NUMERIC_COLS:
        vals = _extract_numeric(train_df[col])
        clip = np.nanpercentile(vals, 97)
        vals = vals.clip(upper=clip)
        num_data.append(vals.fillna(0))

    num_stack = np.column_stack(num_data)

    state["num_mean"] = np.mean(num_stack, axis=0)
    state["num_std"] = np.std(num_stack, axis=0) + 1e-8
    state["num_clip"] = [
        np.nanpercentile(_extract_numeric(train_df[c]), 97)
        for c in NUMERIC_COLS
    ]

    # ---------- MULTI-HOT ----------
    def get_cats(series):
        cats = set()
        for v in series.dropna():
            for t in str(v).split(","):
                cats.add(t.strip())
        return sorted(cats)

    state["room_cats"] = get_cats(train_df[COL_ROOM])
    state["who_cats"] = get_cats(train_df[COL_WHO])
    state["season_cats"] = get_cats(train_df[COL_SEASON])

    # ---------- TF-IDF ----------
    text = (
        train_df[COL_DESC] + " " +
        train_df[COL_FOOD] + " " +
        train_df[COL_SOUNDTRACK]
    )

    vec = TfidfVectorizer(max_features=max_features, min_df=min_df)
    vec.fit(text)

    state["_vectorizer"] = vec

    return state

# ============================================================
# TRANSFORM
# ============================================================

def transform_df(df, state, add_interactions=False):
    n = len(df)

    # ---------- NUMERIC ----------
    num_list = []
    for i, col in enumerate(NUMERIC_COLS):
        vals = _extract_numeric(df[col])

        for p, med in state["num_medians"][col].items():
            mask = (df[COL_TARGET] == p) & vals.isna()
            vals[mask] = med

        vals = vals.clip(upper=state["num_clip"][i])
        vals = vals.fillna(0)

        vals = (vals - state["num_mean"][i]) / state["num_std"][i]
        num_list.append(vals)

    X_num = np.column_stack(num_list)

    # ---------- LIKERT ----------
    likert_list = []
    for col in LIKERT_COLS:
        vals = _extract_likert(df[col])

        for p, med in state["likert_medians"][col].items():
            mask = (df[COL_TARGET] == p) & vals.isna()
            vals[mask] = med

        vals = vals.clip(1, 5).fillna(3)
        likert_list.append(vals)

    X_likert = np.column_stack(likert_list)
    # ---------- INTERACTIONS (OPTIONAL) ----------
    # LIKERT_COLS = [SOMBRE, CONTENT, CALM, UNEASY]
    if add_interactions:
        content_x_calm = (X_likert[:, 1] * X_likert[:, 2]).reshape(-1, 1)
        uneasy_x_sombre = (X_likert[:, 3] * X_likert[:, 0]).reshape(-1, 1)
        X_interactions = np.hstack([content_x_calm, uneasy_x_sombre])
    else:
        X_interactions = None

    # ---------- MULTI-HOT ----------
    def multi_hot(series, cats):
        mat = np.zeros((n, len(cats)))
        idx = {c: i for i, c in enumerate(cats)}

        for i, v in enumerate(series):
            for t in str(v).split(","):
                t = t.strip()
                if t in idx:
                    mat[i, idx[t]] = 1
        return mat

    X_room = multi_hot(df[COL_ROOM], state["room_cats"])
    X_who = multi_hot(df[COL_WHO], state["who_cats"])
    X_season = multi_hot(df[COL_SEASON], state["season_cats"])

    # ---------- TF-IDF ----------
    text = (
        df[COL_DESC] + " " +
        df[COL_FOOD] + " " +
        df[COL_SOUNDTRACK]
    )

    X_text = state["_vectorizer"].transform(text).toarray()

    # ---------- COMBINE ----------
    if X_interactions is None:
        X = np.hstack([X_num, X_likert, X_room, X_who, X_season, X_text])
    else:
        X = np.hstack([X_num, X_likert, X_interactions, X_room, X_who, X_season, X_text])

    y = df[COL_TARGET].map(state["class_to_idx"]).values

    return X, y