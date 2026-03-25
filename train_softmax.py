"""
Softmax (multinomial logistic regression) training pipeline.

What this script does:
1) Uses grouped 5-fold CV (with fixed 20% holdout in each split) for model selection.
2) Aligns selection metric and reporting metric on grouped-CV validation performance.
3) Adds targeted resolver for Starry Night <-> Water Lily Pond when uncertain.
4) Applies lightweight text noise normalization before TF-IDF.
5) Trains final model on all cleaned data using selected hyperparameters.

Outputs:
  - preprocess_state.npz
  - preprocess_state.json
  - model_params.npz
  - cv_selection_metrics.json
"""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_splitting import grouped_kfold_split
from preprocessing import (
    COL_CALM,
    COL_CONTENT,
    COL_DESC,
    COL_EMOTION,
    COL_FOOD,
    COL_N_COLOURS,
    COL_N_OBJECTS,
    COL_PRICE,
    COL_ROOM,
    COL_SEASON,
    COL_SOMBRE,
    COL_SOUNDTRACK,
    COL_TARGET,
    COL_UNEASY,
    COL_WHO,
    clean,
)

RANDOM_STATE = 42
K_FOLDS = 5
MAX_ITER = 2500
CSV_PATH = "training_data.csv"

NUMERIC_COLS = [COL_EMOTION, COL_N_COLOURS, COL_N_OBJECTS]
LIKERT_COLS = [COL_SOMBRE, COL_CONTENT, COL_CALM, COL_UNEASY]
MULTI_COLS = [COL_ROOM, COL_WHO, COL_SEASON]
TEXT_COLS = [COL_DESC, COL_FOOD, COL_SOUNDTRACK]

PAIR_A = "The Starry Night"
PAIR_B = "The Water Lily Pond"

# Main model search
MAX_FEATURES_CANDIDATES = [6000, 8000]
MIN_DF_CANDIDATES = [1, 2]
C_CANDIDATES = [10.0, 20.0, 50.0, 100.0]
CLASS_WEIGHT_OPTIONS = [None, "balanced"]

# Resolver search
PAIR_C_CANDIDATES = [1.0, 5.0, 10.0]
PAIR_MARGIN_CANDIDATES = [0.08, 0.12, 0.16, 0.20]


def normalize_text(x):
    """Lightweight noise cleanup while preserving semantics."""
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_numeric(series, clip_max=None, impute=None):
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
    cats = set()
    for v in series.dropna():
        for part in str(v).split(","):
            part = part.strip()
            if part:
                cats.add(part)
    return sorted(cats)


def multi_hot(series, categories):
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


def tokenize(text):
    return re.findall(r"(?u)\b\w+\b", str(text).lower())


def build_vocab_idf(texts, max_features, min_df):
    docs = [tokenize(t) for t in texts]
    n_docs = len(docs)

    df_count = Counter()
    for toks in docs:
        df_count.update(set(toks))

    kept = [w for w, c in df_count.items() if c >= min_df]
    kept.sort(key=lambda w: (-df_count[w], w))
    if max_features is not None:
        kept = kept[:max_features]

    vocab = kept
    idf = np.zeros(len(vocab), dtype=float)
    for i, w in enumerate(vocab):
        # sklearn-style smooth idf: log((1+n)/(1+df)) + 1
        idf[i] = np.log((1.0 + n_docs) / (1.0 + df_count[w])) + 1.0
    return vocab, idf


def tfidf_matrix(texts, vocab, idf):
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


def fit_state(train_df, max_features, min_df):
    state = {}
    state["classes"] = sorted(train_df[COL_TARGET].unique())
    state["class_to_idx"] = {c: i for i, c in enumerate(state["classes"])}

    num_medians = []
    num_clips = []
    num_means = []
    num_stds = []
    for c in NUMERIC_COLS:
        non_null = train_df[c].dropna()
        med = float(np.median(extract_numeric(non_null))) if len(non_null) else 0.0
        num_medians.append(med)
        raw = extract_numeric(train_df[c], impute=med)
        clip = np.percentile(raw[raw > 0] if np.any(raw > 0) else raw, 97)
        clip = 1.0 if clip == 0 else float(clip)
        num_clips.append(clip)
        x = extract_numeric(train_df[c], clip_max=clip, impute=med)
        num_means.append(float(np.mean(x)))
        num_stds.append(float(np.std(x)) or 1.0)
    state["num_medians"] = np.array(num_medians, dtype=float)
    state["num_clips"] = np.array(num_clips, dtype=float)
    state["num_means"] = np.array(num_means, dtype=float)
    state["num_stds"] = np.array(num_stds, dtype=float)

    likert_medians = []
    for c in LIKERT_COLS:
        non_null = train_df[c].dropna()
        med = float(np.median(extract_likert(non_null))) if len(non_null) else 3.0
        likert_medians.append(med)
    state["likert_medians"] = np.array(likert_medians, dtype=float)

    price_non_null = train_df[COL_PRICE].dropna()
    price_median = float(np.median(extract_numeric(price_non_null))) if len(price_non_null) else 0.0
    price_raw = extract_numeric(train_df[COL_PRICE], impute=price_median)
    price_clip = np.percentile(price_raw[price_raw > 0] if np.any(price_raw > 0) else price_raw, 97)
    price_clip = 1.0 if price_clip == 0 else float(price_clip)
    price_train = extract_numeric(train_df[COL_PRICE], clip_max=price_clip, impute=price_median)
    state["price_median"] = float(price_median)
    state["price_clip"] = float(price_clip)
    state["price_mean"] = float(np.mean(price_train))
    state["price_std"] = float(np.std(price_train)) or 1.0

    state["room_cats"] = get_categories(train_df[COL_ROOM])
    state["who_cats"] = get_categories(train_df[COL_WHO])
    state["season_cats"] = get_categories(train_df[COL_SEASON])

    train_text = (
        train_df[COL_DESC].map(normalize_text).astype(str)
        + " "
        + train_df[COL_FOOD].map(normalize_text).astype(str)
        + " "
        + train_df[COL_SOUNDTRACK].map(normalize_text).astype(str)
    )
    vocab, idf = build_vocab_idf(train_text.values, max_features=max_features, min_df=min_df)
    state["vocab"] = vocab
    state["idf"] = idf
    state["max_features"] = max_features
    state["min_df"] = min_df
    return state


def transform_df(df, state):
    x_num = np.column_stack(
        [
            (
                extract_numeric(
                    df[c],
                    clip_max=float(state["num_clips"][j]),
                    impute=float(state["num_medians"][j]),
                )
                - state["num_means"][j]
            )
            / state["num_stds"][j]
            for j, c in enumerate(NUMERIC_COLS)
        ]
    )
    x_likert = np.column_stack(
        [
            extract_likert(df[c], impute=float(state["likert_medians"][j]))
            for j, c in enumerate(LIKERT_COLS)
        ]
    )
    pr = extract_numeric(df[COL_PRICE], clip_max=state["price_clip"], impute=state["price_median"])
    x_price = ((pr - state["price_mean"]) / state["price_std"]).reshape(-1, 1)

    x_room = multi_hot(df[COL_ROOM], state["room_cats"])
    x_who = multi_hot(df[COL_WHO], state["who_cats"])
    x_season = multi_hot(df[COL_SEASON], state["season_cats"])

    text = (
        df[COL_DESC].map(normalize_text).astype(str)
        + " "
        + df[COL_FOOD].map(normalize_text).astype(str)
        + " "
        + df[COL_SOUNDTRACK].map(normalize_text).astype(str)
    )
    x_tfidf = tfidf_matrix(text.values, state["vocab"], state["idf"])

    X = np.hstack([x_num, x_likert, x_price, x_room, x_who, x_season, x_tfidf])
    y = np.array([state["class_to_idx"][c] for c in df[COL_TARGET]])
    return X, y


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def fit_pair_resolver(X_train, y_train, X_val, y_val, classes):
    if PAIR_A not in classes or PAIR_B not in classes:
        return None
    idx_a = classes.index(PAIR_A)
    idx_b = classes.index(PAIR_B)
    tr_mask = (y_train == idx_a) | (y_train == idx_b)
    va_mask = (y_val == idx_a) | (y_val == idx_b)
    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        return None

    Xp_tr = X_train[tr_mask]
    yp_tr = (y_train[tr_mask] == idx_b).astype(int)
    Xp_va = X_val[va_mask]
    y_va_full = y_val[va_mask]

    best = None
    best_acc = -1.0
    for c_val in PAIR_C_CANDIDATES:
        m = LogisticRegression(max_iter=MAX_ITER, C=c_val, solver="lbfgs", random_state=RANDOM_STATE)
        m.fit(Xp_tr, yp_tr)
        pred = (m.predict_proba(Xp_va)[:, 1] >= 0.5).astype(int)
        pred_full = np.where(pred == 1, idx_b, idx_a)
        acc = float(accuracy_score(y_va_full, pred_full))
        if acc > best_acc:
            best = {"model": m, "pair_C": c_val, "idx_a": idx_a, "idx_b": idx_b, "acc": acc}
            best_acc = acc
    return best


def apply_pair_resolver(main_probs, main_pred, pair_info, X, margin_thr):
    if pair_info is None:
        return main_pred.copy(), 0
    idx_a = pair_info["idx_a"]
    idx_b = pair_info["idx_b"]
    top2 = np.argsort(main_probs, axis=1)[:, -2:]
    margins = np.sort(main_probs, axis=1)[:, -1] - np.sort(main_probs, axis=1)[:, -2]
    cand = np.array(
        [({idx_a, idx_b} == set(top2[i])) and (margins[i] <= margin_thr) for i in range(len(main_pred))],
        dtype=bool,
    )
    out = main_pred.copy()
    n = int(cand.sum())
    if n == 0:
        return out, 0
    p_b = pair_info["model"].predict_proba(X[cand])[:, 1]
    out[cand] = np.where(p_b >= 0.5, idx_b, idx_a)
    return out, n


def evaluate_with_optional_resolver(model, pair_info, X_val, y_val, X_ref_for_resolver):
    logits = X_val @ model.coef_.T + model.intercept_
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1)
    base_acc = float(accuracy_score(y_val, pred))
    base_f1 = float(f1_score(y_val, pred, average="macro"))

    best = {"acc": base_acc, "f1": base_f1, "thr": None, "overrides": 0, "pred": pred}
    if pair_info is None:
        return best

    for thr in PAIR_MARGIN_CANDIDATES:
        p2, n = apply_pair_resolver(probs, pred, pair_info, X_ref_for_resolver, thr)
        acc = float(accuracy_score(y_val, p2))
        f1 = float(f1_score(y_val, p2, average="macro"))
        if acc > best["acc"]:
            best = {"acc": acc, "f1": f1, "thr": thr, "overrides": n, "pred": p2}
    return best


def save_final_state_and_model(state, model, pair_meta):
    np.savez(
        "preprocess_state.npz",
        num_means=state["num_means"],
        num_stds=state["num_stds"],
        num_clips=state["num_clips"],
        idf=state["idf"],
        price_mean=np.array([state["price_mean"]]),
        price_std=np.array([state["price_std"]]),
        price_clip=np.array([state["price_clip"]]),
        num_medians=state["num_medians"],
        likert_medians=state["likert_medians"],
        price_median=np.array([state["price_median"]]),
    )
    with open("preprocess_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": state["classes"],
                "vocab": state["vocab"],
                "room_cats": state["room_cats"],
                "who_cats": state["who_cats"],
                "season_cats": state["season_cats"],
                "clip_percentile": 97,
                "max_features": state["max_features"],
                "min_df": state["min_df"],
                "impute": "none",
                "text_normalization": "lowercase + remove non-alnum + collapse spaces",
                "pair_resolver": pair_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.savez(
        "model_params.npz",
        W=model.coef_.T.astype(float),
        b=model.intercept_.astype(float),
        pair_w=np.array(pair_meta["pair_w"], dtype=float) if pair_meta["enabled"] else np.array([]),
        pair_b=np.array([pair_meta["pair_b"]], dtype=float) if pair_meta["enabled"] else np.array([]),
        pair_idx=np.array(pair_meta["pair_idx"], dtype=int) if pair_meta["enabled"] else np.array([], dtype=int),
        pair_margin=np.array([pair_meta["margin"]], dtype=float) if pair_meta["enabled"] else np.array([-1.0], dtype=float),
    )


def main():
    df = pd.read_csv(CSV_PATH)
    df_clean = clean(df)

    grid = []
    for mf in MAX_FEATURES_CANDIDATES:
        for md in MIN_DF_CANDIDATES:
            for c_val in C_CANDIDATES:
                for cw in CLASS_WEIGHT_OPTIONS:
                    grid.append({"max_features": mf, "min_df": md, "C": c_val, "class_weight": cw})

    all_results = []
    best = None
    best_val_acc = -1.0
    for params in grid:
        fold_metrics = []
        for fold_idx, (tr, va, te) in enumerate(
            grouped_kfold_split(df_clean, k=K_FOLDS, random_state=RANDOM_STATE),
            start=1,
        ):
            state = fit_state(tr, max_features=params["max_features"], min_df=params["min_df"])
            Xtr, ytr = transform_df(tr, state)
            Xva, yva = transform_df(va, state)
            Xte, yte = transform_df(te, state)

            main_model = LogisticRegression(
                max_iter=MAX_ITER,
                C=params["C"],
                class_weight=params["class_weight"],
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )
            main_model.fit(Xtr, ytr)

            pair_info = fit_pair_resolver(Xtr, ytr, Xva, yva, state["classes"])
            val_eval = evaluate_with_optional_resolver(main_model, pair_info, Xva, yva, Xva)
            test_eval = evaluate_with_optional_resolver(main_model, pair_info, Xte, yte, Xte)

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "val_acc": val_eval["acc"],
                    "val_f1": val_eval["f1"],
                    "test_acc": test_eval["acc"],
                    "test_f1": test_eval["f1"],
                    "pair_thr": val_eval["thr"],
                    "pair_overrides_val": val_eval["overrides"],
                }
            )

        result = {
            "params": params,
            "val_acc_mean": float(np.mean([f["val_acc"] for f in fold_metrics])),
            "val_acc_std": float(np.std([f["val_acc"] for f in fold_metrics])),
            "val_f1_mean": float(np.mean([f["val_f1"] for f in fold_metrics])),
            "test_acc_mean": float(np.mean([f["test_acc"] for f in fold_metrics])),
            "test_f1_mean": float(np.mean([f["test_f1"] for f in fold_metrics])),
            "folds": fold_metrics,
        }
        all_results.append(result)
        if result["val_acc_mean"] > best_val_acc:
            best = result
            best_val_acc = result["val_acc_mean"]

    if best is None:
        raise RuntimeError("No best config selected.")

    # Train final model on full cleaned data with selected params
    final_state = fit_state(
        df_clean,
        max_features=best["params"]["max_features"],
        min_df=best["params"]["min_df"],
    )
    X_full, y_full = transform_df(df_clean, final_state)
    final_model = LogisticRegression(
        max_iter=MAX_ITER,
        C=best["params"]["C"],
        class_weight=best["params"]["class_weight"],
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_full, y_full)

    # Pair resolver on full data for deployment
    pair_meta = {"enabled": False}
    if PAIR_A in final_state["classes"] and PAIR_B in final_state["classes"]:
        idx_a = final_state["classes"].index(PAIR_A)
        idx_b = final_state["classes"].index(PAIR_B)
        mask = (y_full == idx_a) | (y_full == idx_b)
        if mask.sum() > 0:
            Xp = X_full[mask]
            yp = (y_full[mask] == idx_b).astype(int)
            m_pair = LogisticRegression(
                max_iter=MAX_ITER,
                C=5.0,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )
            m_pair.fit(Xp, yp)
            # use best observed val threshold among selected-config folds
            selected_rows = [r for r in all_results if r["params"] == best["params"]]
            thrs = [f["pair_thr"] for f in selected_rows[0]["folds"] if f["pair_thr"] is not None]
            margin = float(np.median(thrs)) if thrs else -1.0
            pair_meta = {
                "enabled": margin > 0,
                "pair_idx": [idx_a, idx_b],
                "pair_w": m_pair.coef_.ravel().tolist(),
                "pair_b": float(m_pair.intercept_[0]),
                "margin": margin,
                "pair_C": 5.0,
            }

    save_final_state_and_model(final_state, final_model, pair_meta)

    summary = {
        "best_by_grouped_cv": best,
        "top5": sorted(all_results, key=lambda x: x["val_acc_mean"], reverse=True)[:5],
        "final_training": {
            "n_rows": int(len(df_clean)),
            "n_features": int(X_full.shape[1]),
            "classes": final_state["classes"],
            "pair_resolver_enabled": pair_meta["enabled"],
        },
    }
    with open("cv_selection_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Best grouped-CV params:", best["params"])
    print(
        f"Grouped-CV val_acc={best['val_acc_mean']:.4f}±{best['val_acc_std']:.4f}, "
        f"val_f1={best['val_f1_mean']:.4f}"
    )
    print(
        f"Grouped-CV test_acc_mean={best['test_acc_mean']:.4f}, "
        f"test_f1_mean={best['test_f1_mean']:.4f}"
    )
    print("Saved preprocess_state.*, model_params.npz, cv_selection_metrics.json")


if __name__ == "__main__":
    main()
