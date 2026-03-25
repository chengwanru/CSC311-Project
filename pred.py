"""
Prediction script aligned with train_softmax.py.

Allowed imports: standard library, numpy, pandas.
Implements:
  - text normalization
  - softmax main classifier
  - optional Starry/Lily pair resolver on uncertain predictions
"""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd

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


def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x).lower().replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_numeric(series, clip_max=None, impute=None):
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


def _extract_likert(series, impute=None):
    fill = int(round(np.clip(float(impute), 1, 5))) if impute is not None else 0
    out = []
    for v in series:
        if pd.isna(v):
            out.append(fill)
            continue
        m = re.search(r"^([1-5])", str(v).strip())
        out.append(int(m.group(1)) if m else fill)
    return np.array(out, dtype=float)


def _multi_hot(series, categories):
    n = len(series)
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((n, k), dtype=float)
    for i, v in enumerate(series):
        if pd.isna(v):
            continue
        for part in str(v).split(","):
            key = part.strip()
            if key in cat_to_idx:
                mat[i, cat_to_idx[key]] = 1.0
    return mat


def _tokenize(text):
    return re.findall(r"(?u)\b\w+\b", str(text).lower())


def _tfidf_matrix(texts, vocab, idf):
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, t in enumerate(texts):
        counts = Counter(_tokenize(t))
        for tok, cnt in counts.items():
            j = vocab_to_idx.get(tok)
            if j is not None:
                X[i, j] = float(cnt) * float(idf[j])
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


def _softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _apply_pair_resolver_if_enabled(probs, X, mp):
    pair_idx = mp["pair_idx"] if "pair_idx" in mp else np.array([], dtype=int)
    pair_margin = float(mp["pair_margin"][0]) if "pair_margin" in mp else -1.0
    if len(pair_idx) != 2 or pair_margin <= 0:
        return np.argmax(probs, axis=1)

    idx_a, idx_b = int(pair_idx[0]), int(pair_idx[1])
    pair_w = mp["pair_w"] if "pair_w" in mp else np.array([])
    pair_b = float(mp["pair_b"][0]) if "pair_b" in mp and len(mp["pair_b"]) > 0 else 0.0
    if len(pair_w) == 0:
        return np.argmax(probs, axis=1)

    pred = np.argmax(probs, axis=1)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    margins = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]

    cand = np.array(
        [({idx_a, idx_b} == set(top2[i])) and (margins[i] <= pair_margin) for i in range(len(pred))],
        dtype=bool,
    )
    if not np.any(cand):
        return pred

    logits_pair = X[cand] @ pair_w.reshape(-1, 1) + pair_b
    p_b = 1.0 / (1.0 + np.exp(-logits_pair.ravel()))
    pred[cand] = np.where(p_b >= 0.5, idx_b, idx_a)
    return pred


def predict_all(csv_filename):
    df = pd.read_csv(csv_filename)
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("")
    for c in MULTI_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("")

    with open("preprocess_state.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    st = np.load("preprocess_state.npz", allow_pickle=False)
    mp = np.load("model_params.npz", allow_pickle=False)

    classes = cfg["classes"]
    vocab = cfg["vocab"]
    room_cats = cfg["room_cats"]
    who_cats = cfg["who_cats"]
    season_cats = cfg["season_cats"]

    num_means = st["num_means"]
    num_stds = st["num_stds"]
    num_clips = st["num_clips"]
    idf = st["idf"]
    price_mean = float(st["price_mean"][0])
    price_std = float(st["price_std"][0])
    price_clip = float(st["price_clip"][0])
    num_medians = st["num_medians"] if len(st["num_medians"]) > 0 else None
    likert_medians = st["likert_medians"] if len(st["likert_medians"]) > 0 else None
    price_median = float(st["price_median"][0]) if len(st["price_median"]) > 0 else None

    x_num = np.column_stack(
        [
            (
                _extract_numeric(
                    df[c],
                    clip_max=float(num_clips[j]),
                    impute=float(num_medians[j]) if num_medians is not None else None,
                )
                - num_means[j]
            )
            / num_stds[j]
            for j, c in enumerate(NUMERIC_COLS)
        ]
    )
    x_likert = np.column_stack(
        [
            _extract_likert(
                df[c],
                impute=float(likert_medians[j]) if likert_medians is not None else None,
            )
            for j, c in enumerate(LIKERT_COLS)
        ]
    )
    pr = _extract_numeric(df[COL_PRICE], clip_max=price_clip, impute=price_median)
    x_price = ((pr - price_mean) / price_std).reshape(-1, 1)

    x_room = _multi_hot(df[COL_ROOM], room_cats)
    x_who = _multi_hot(df[COL_WHO], who_cats)
    x_season = _multi_hot(df[COL_SEASON], season_cats)

    text = (
        df[COL_DESC].map(normalize_text).astype(str)
        + " "
        + df[COL_FOOD].map(normalize_text).astype(str)
        + " "
        + df[COL_SOUNDTRACK].map(normalize_text).astype(str)
    )
    x_tfidf = _tfidf_matrix(text.values, vocab, idf)

    X = np.hstack([x_num, x_likert, x_price, x_room, x_who, x_season, x_tfidf])
    probs = _softmax(X @ mp["W"] + mp["b"])
    pred_idx = _apply_pair_resolver_if_enabled(probs, X, mp)
    return [classes[i] for i in pred_idx]
