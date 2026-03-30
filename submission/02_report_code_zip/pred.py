"""
CSC311 MarkUs prediction script.

Imports allowed: standard library, numpy, pandas only (no sklearn).

Requires in the same directory:
  model_state.json
  model_weights.npz

Generate them with:  python export_model.py
(training uses sklearn; this file loads exported weights only.)
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Column names (must match pipeline.training_data.csv) ─────────────────────
_COL_ID = "unique_id"
_COL_TARGET = "Painting"
_COL_EMOTION = (
    "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?"
)
_COL_DESC = "Describe how this painting makes you feel."
_COL_SOMBRE = "This art piece makes me feel sombre."
_COL_CONTENT = "This art piece makes me feel content."
_COL_CALM = "This art piece makes me feel calm."
_COL_UNEASY = "This art piece makes me feel uneasy."
_COL_N_COLOURS = "How many prominent colours do you notice in this painting?"
_COL_N_OBJECTS = "How many objects caught your eye in the painting?"
_COL_PRICE = "How much (in Canadian dollars) would you be willing to pay for this painting?"
_COL_ROOM = "If you could purchase this painting, which room would you put that painting in?"
_COL_WHO = "If you could view this art in person, who would you want to view it with?"
_COL_SEASON = "What season does this art piece remind you of?"
_COL_FOOD = "If this painting was a food, what would be?"
_COL_SOUNDTRACK = (
    "Imagine a soundtrack for this painting. "
    "Describe that soundtrack without naming any objects in the painting."
)
_NUMERIC_COLS = [_COL_EMOTION, _COL_N_COLOURS, _COL_N_OBJECTS]
_LIKERT_COLS = [_COL_SOMBRE, _COL_CONTENT, _COL_CALM, _COL_UNEASY]
_MULTI_COLS = [_COL_ROOM, _COL_WHO, _COL_SEASON]
_TEXT_COLS = [_COL_DESC, _COL_FOOD, _COL_SOUNDTRACK]
_REQUIRED_CLEAN = (
    [_COL_ID, _COL_TARGET, _COL_EMOTION, _COL_N_COLOURS, _COL_N_OBJECTS] + _LIKERT_COLS
)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=_REQUIRED_CLEAN).copy()
    for c in _TEXT_COLS:
        if c in d.columns:
            d[c] = d[c].fillna("")
    for c in _MULTI_COLS:
        if c in d.columns:
            d[c] = d[c].fillna("")
    return d


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


def _get_categories(series):
    cats = set()
    for v in series.dropna():
        for part in str(v).split(","):
            p = part.strip()
            if p:
                cats.add(p)
    return sorted(cats)


def _multi_hot(series, categories):
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


def _normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x).lower().replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize_lr(text):
    return re.findall(r"(?u)\b\w+\b", str(text).lower())


def _tfidf_matrix(texts, vocab, idf):
    v2i = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, t in enumerate(texts):
        counts = Counter(_tokenize_lr(t))
        for tok, cnt in counts.items():
            j = v2i.get(tok)
            if j is not None:
                X[i, j] = float(cnt) * float(idf[j])
        nrm = np.linalg.norm(X[i])
        if nrm > 0:
            X[i] /= nrm
    return X


def _lr_X(df: pd.DataFrame, st: Dict[str, Any]) -> np.ndarray:
    x_num = np.column_stack(
        [
            (
                _extract_numeric(
                    df[c],
                    clip_max=float(st["num_clips"][j]),
                    impute=float(st["num_medians"][j]),
                )
                - float(st["num_means"][j])
            )
            / float(st["num_stds"][j])
            for j, c in enumerate(_NUMERIC_COLS)
        ]
    )
    x_likert = np.column_stack(
        [
            _extract_likert(df[c], impute=float(st["likert_medians"][j]))
            for j, c in enumerate(_LIKERT_COLS)
        ]
    )
    pr = _extract_numeric(
        df[_COL_PRICE],
        clip_max=float(st["price_clip"]),
        impute=float(st["price_median"]),
    )
    x_price = ((pr - float(st["price_mean"])) / float(st["price_std"])).reshape(-1, 1)
    x_room = _multi_hot(df[_COL_ROOM], st["room_cats"])
    x_who = _multi_hot(df[_COL_WHO], st["who_cats"])
    x_season = _multi_hot(df[_COL_SEASON], st["season_cats"])
    text = (
        df[_COL_DESC].map(_normalize_text).astype(str)
        + " "
        + df[_COL_FOOD].map(_normalize_text).astype(str)
        + " "
        + df[_COL_SOUNDTRACK].map(_normalize_text).astype(str)
    )
    x_tfidf = _tfidf_matrix(text.values, st["vocab"], np.array(st["idf"], dtype=float))
    return np.hstack([x_num, x_likert, x_price, x_room, x_who, x_season, x_tfidf])


_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "it", "this", "that", "with", "as", "was", "are", "be", "i", "me",
    "my", "we", "not", "have", "had", "has", "do", "does", "did", "by", "from",
    "its", "into", "so", "if", "no", "up", "out", "about", "which", "like",
    "very", "just", "would", "could", "should", "there", "their", "they",
    "when", "what", "how", "all", "some", "any", "can", "more", "been",
    "also", "will", "than", "then", "here", "make", "feel", "feels", "made",
    "makes",
    "painting", "art", "picture", "artwork", "image", "piece",
    "pizza", "bread", "water", "nothing", "nan",
}


def _stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for suffix in (
        "ingly", "ness", "ment", "ful", "less", "ing", "tion", "sion", "ous",
        "ive", "ize", "ise", "est", "er", "ed", "ly", "es", "s",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: -len(suffix)]
    return word


def _tokenize_nb(text):
    if not text or pd.isna(text):
        return []
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in (_stem(t) for t in tokens) if t not in _STOP_WORDS and len(t) > 1]


def _build_vocab_nb(texts, vocab_size):
    freq = defaultdict(int)
    for text in texts:
        for tok in _tokenize_nb(text):
            freq[tok] += 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])]
    return vocab[:vocab_size]


def _texts_to_matrix(texts, vocab, binary=False):
    wi = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row_i, text in enumerate(texts):
        for tok in _tokenize_nb(text):
            if tok in wi:
                mat[row_i, wi[tok]] += 1
    if binary:
        mat = (mat > 0).astype(np.float32)
    return mat


def _encode_multi(series, known_values):
    rows = []
    for cell in series:
        active = set()
        if not pd.isna(cell):
            for v in str(cell).split(","):
                active.add(v.strip())
        rows.append({v: int(v in active) for v in known_values})
    return pd.DataFrame(rows, index=series.index)


_MULTI_KEY = {_COL_ROOM: "room", _COL_WHO: "companion", _COL_SEASON: "season"}
_NUM_COLS_NB = [_COL_EMOTION, _COL_N_COLOURS, _COL_N_OBJECTS]
_P99_COLS = [_COL_N_COLOURS, _COL_N_OBJECTS]
_PRICE_NB = _COL_PRICE


def _build_nb_X(
    df: pd.DataFrame, info: Dict[str, Any], params: Dict[str, Any]
) -> np.ndarray:
    d = df.copy()
    parts = []
    combined_text = d[_TEXT_COLS].fillna("").agg(" ".join, axis=1)
    vocab = info["vocab"]
    text_mat = _texts_to_matrix(combined_text, vocab, binary=params["binary_tf"])
    parts.append(text_mat)
    likert_mat = np.column_stack(
        [
            _extract_likert(d[c], impute=info["likert_medians"][i])
            for i, c in enumerate(_LIKERT_COLS)
        ]
    ).astype(np.float32)
    parts.append(likert_mat)
    num_cols_parsed = []
    for c in _NUM_COLS_NB:
        clip_hi = info["numeric_p99_upper"].get(c)
        num_cols_parsed.append(_extract_numeric(d[c], clip_max=clip_hi))
    num_cols_parsed.append(np.log1p(_extract_numeric(d[_PRICE_NB])))
    num_arr = np.column_stack(num_cols_parsed).astype(np.float32)
    for j in range(num_arr.shape[1]):
        zm = num_arr[:, j] == 0
        num_arr[zm, j] = float(info["num_medians"][j])
    num_arr = (num_arr - np.array(info["num_mean"], dtype=np.float32)) / np.array(
        info["num_std"], dtype=np.float32
    )
    parts.append(num_arr)
    multi_parts = []
    for col in _MULTI_COLS:
        kv_key = f"multi_{_MULTI_KEY[col]}"
        enc = _encode_multi(d[col], info[kv_key])
        multi_parts.append(enc.values.astype(np.float32))
    parts.append(np.hstack(multi_parts))
    return np.hstack(parts)


def _apply_weights_nb(X: np.ndarray, params: Dict[str, Any], block_cols: Dict) -> np.ndarray:
    X = X.copy().astype(np.float32)
    for block, w in [
        ("likert", params["w_likert"]),
        ("numeric", params["w_numeric"]),
        ("cat", params["w_categorical"]),
    ]:
        start, end = block_cols[block]
        X[:, start:end] *= w
    price_col = block_cols["numeric"][1] - 1
    X[:, price_col] *= params.get("w_price", 0.0)
    return X


def _nb_lp_from_arrays(
    X: np.ndarray,
    log_priors,
    log_likelihoods,
    mn_idx,
    g_idx,
    gauss_mean,
    gauss_std,
) -> np.ndarray:
    mn_idx = np.asarray(mn_idx, dtype=int)
    g_idx = np.asarray(g_idx, dtype=int)
    X_mn = X[:, mn_idx]
    log_lik = X_mn @ log_likelihoods.T
    if len(g_idx) > 0:
        X_g = X[:, g_idx]
        n = X_g.shape[0]
        k = log_priors.shape[0]
        gpart = np.zeros((n, k))
        for kk in range(k):
            mu = gauss_mean[kk]
            sig = gauss_std[kk]
            gpart[:, kk] = -0.5 * np.sum(
                np.log(2 * np.pi * sig**2) + ((X_g - mu) / sig) ** 2, axis=1
            )
        log_lik = log_lik + gpart
    return log_lik + log_priors


def _blend_nb_cnb_lp(X, w, nb_blend: float):
    nb_lp = _nb_lp_from_arrays(
        X,
        w["nb_log_priors"],
        w["nb_log_likelihoods"],
        w["nb_mn_idx"],
        w["nb_g_idx"],
        w["nb_gauss_mean"],
        w["nb_gauss_std"],
    )
    cnb_lp = _nb_lp_from_arrays(
        X,
        w["cnb_log_priors"],
        w["cnb_log_likelihoods"],
        w["cnb_mn_idx"],
        w["cnb_g_idx"],
        w["cnb_gauss_mean"],
        w["cnb_gauss_std"],
    )
    nb_lp = nb_lp - nb_lp.max(axis=1, keepdims=True)
    cnb_lp = cnb_lp - cnb_lp.max(axis=1, keepdims=True)
    blended = nb_blend * nb_lp + (1.0 - nb_blend) * cnb_lp
    e = np.exp(blended - blended.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _softmax_lr(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _lr_proba(X: np.ndarray, w) -> np.ndarray:
    logits = X @ w["lr_coef"].T + w["lr_intercept"]
    return _softmax_lr(logits)


def _meta_predict_proba(meta_x: np.ndarray, w) -> np.ndarray:
    logits = meta_x @ w["meta_coef"].T + w["meta_intercept"]
    return _softmax_lr(logits)


def _tree_proba_row(x, cl, cr, feat, thr, val, n_active: int) -> np.ndarray:
    node = 0
    while node < n_active and int(cl[node]) != -1:
        f = int(feat[node])
        if f < 0 or f >= len(x):
            break
        if x[f] <= thr[node]:
            node = int(cl[node])
        else:
            node = int(cr[node])
    vv = val[node].astype(float)
    s = float(vv.sum())
    return vv / s if s > 0 else np.ones(3) / 3.0


def _rf_proba(X: np.ndarray, w) -> np.ndarray:
    n_trees = int(w["rf_n_trees"][0])
    n_samples = X.shape[0]
    acc = np.zeros((n_samples, 3))
    for ti in range(n_trees):
        cl = w[f"rf{ti}_cl"]
        cr = w[f"rf{ti}_cr"]
        feat = w[f"rf{ti}_feat"]
        thr = w[f"rf{ti}_thr"]
        val = w[f"rf{ti}_val"]
        na = int(w["rf_active_nodes"][ti])
        for i in range(n_samples):
            acc[i] += _tree_proba_row(X[i], cl, cr, feat, thr, val, na)
    return acc / max(n_trees, 1)


def predict_all(csv_path: str) -> List[str]:
    """
    Load a CSV (same schema as training_data.csv), return predicted Painting labels.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "model_state.json"), encoding="utf-8") as f:
        bundle = json.load(f)
    raw = np.load(os.path.join(root, "model_weights.npz"))
    w = {k: raw[k] for k in raw.files}

    classes: List[str] = bundle["classes"]

    df = pd.read_csv(csv_path)
    if _COL_TARGET not in df.columns:
        df[_COL_TARGET] = classes[0]
    df = _clean(df)
    df = df.sort_values([_COL_ID, _COL_TARGET]).reset_index(drop=True)

    st = bundle["fit_state"]
    X_lr = _lr_X(df, st).astype(np.float64)

    info = bundle["nb_fit_info"]
    params = bundle["nb_feat_params"]
    block_cols = {k: tuple(v) for k, v in info["block_cols"].items()}
    X_nb = _build_nb_X(df, info, params)
    X_nb = _apply_weights_nb(X_nb, params, block_cols).astype(np.float64)

    p_lr = _lr_proba(X_lr, w)
    p_nb = _blend_nb_cnb_lp(X_nb, w, float(bundle["nb_blend"]))
    p_rf = _rf_proba(X_lr, w)
    meta_x = np.hstack([p_lr, p_nb, p_rf])
    proba = _meta_predict_proba(meta_x, w)
    idx = np.argmax(proba, axis=1)
    return [classes[j] for j in idx]


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "training_data.csv"
    preds = predict_all(p)
    print(f"n={len(preds)}  first5={preds[:5]}")
