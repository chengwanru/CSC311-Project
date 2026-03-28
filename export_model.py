"""
Train final stacking on ALL of training_data.csv and export artifacts for pred.py
(sklearn allowed here; pred.py uses only numpy/pandas per course rules).

Run from repo root (with training_data.csv present):
  python export_model.py

Writes: model_state.json, model_weights.npz
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from naive_bayes import (
    NaiveBayes,
    CNaiveBayes,
    apply_weights,
    build_features,
    gaussian_col_indices,
)
from pipeline import CSV_PATH, clean, fit_state, transform_df

# ── Final chosen hyperparameters (multiseed quick report) ────────────────────
RANDOM_STATE = 42
K_FOLDS = 5
MAX_FEATURES = 6000
MIN_DF = 1
LR_C = 100.0
NB_ALPHA = 0.9
NB_BLEND = 1.0
META_C = 0.5
RF_N_EST = 200
RF_DEPTH = None
RF_MIN_LEAF = 1

NB_FEAT_PARAMS = {
    "vocab_size": 2000,
    "alpha": 1.0,
    "binary_tf": True,
    "num_bins": None,
    "blend_weight": 1.0,
    "w_likert": 1.0,
    "w_numeric": 0.5,
    "w_price": 0.0,
    "w_categorical": 1.0,
}

SORT_KEY = ["unique_id", "Painting"]


def _nb_proba(X, nb_m, cnb_m, blend):
    nb_lp = nb_m.predict_log_proba(X)
    cnb_lp = cnb_m.predict_log_proba(X)
    nb_lp -= nb_lp.max(axis=1, keepdims=True)
    cnb_lp -= cnb_lp.max(axis=1, keepdims=True)
    blended = blend * nb_lp + (1.0 - blend) * cnb_lp
    e = np.exp(blended - blended.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _serialize_state(state: dict) -> dict:
    """JSON-serializable view of fit_state dict (no ndarray)."""
    out = {}
    for k, v in state.items():
        if k == "classes":
            out[k] = list(v)
        elif k == "class_to_idx":
            out[k] = {str(a): int(b) for a, b in v.items()}
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = v
    return out


def _export_forest(rf: RandomForestClassifier):
    trees = []
    max_n = 0
    for est in rf.estimators_:
        t = est.tree_
        max_n = max(max_n, t.node_count)
    for est in rf.estimators_:
        t = est.tree_
        n = t.node_count
        cl = np.full(max_n, -1, dtype=np.int32)
        cr = np.full(max_n, -1, dtype=np.int32)
        feat = np.full(max_n, -2, dtype=np.int32)
        thr = np.zeros(max_n, dtype=np.float64)
        val = np.zeros((max_n, 3), dtype=np.float64)
        cl[:n] = t.children_left
        cr[:n] = t.children_right
        feat[:n] = t.feature
        thr[:n] = t.threshold
        val[:n, :] = t.value[:, 0, :]
        trees.append((cl, cr, feat, thr, val, n))
    return trees, max_n


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df_clean = clean(pd.read_csv(CSV_PATH))
    df_clean = df_clean.sort_values(SORT_KEY).reset_index(drop=True)

    state = fit_state(df_clean, max_features=MAX_FEATURES, min_df=MIN_DF)
    X_all, y_all = transform_df(df_clean, state)
    class_to_idx = state["class_to_idx"]
    classes = list(state["classes"])
    K = len(classes)

    _, nb_fit_info = build_features(df_clean, fit_info=None, params=NB_FEAT_PARAMS)
    X_nb_all, _ = build_features(df_clean, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_all = apply_weights(X_nb_all, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    g_cols = gaussian_col_indices(NB_FEAT_PARAMS)

    n = len(df_clean)
    oof_lr = np.zeros((n, K))
    oof_nb = np.zeros((n, K))
    oof_rf = np.zeros((n, K))

    rng = np.random.RandomState(RANDOM_STATE)
    uids = df_clean["unique_id"].unique()
    rng.shuffle(uids)
    folds = np.array_split(uids, K_FOLDS)

    for fi in range(K_FOLDS):
        vset = set(folds[fi])
        tset = set(np.concatenate([folds[j] for j in range(K_FOLDS) if j != fi]))
        tr_m = df_clean["unique_id"].isin(tset).values
        vl_m = df_clean["unique_id"].isin(vset).values

        lr = LogisticRegression(
            max_iter=2500,
            C=LR_C,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
        lr.fit(X_all[tr_m], y_all[tr_m])
        oof_lr[vl_m] = lr.predict_proba(X_all[vl_m])

        nb = NaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
        cnb = CNaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
        nb.fit(X_nb_all[tr_m], y_all[tr_m])
        cnb.fit(X_nb_all[tr_m], y_all[tr_m])
        oof_nb[vl_m] = _nb_proba(X_nb_all[vl_m], nb, cnb, NB_BLEND)

        rf = RandomForestClassifier(
            n_estimators=RF_N_EST,
            max_depth=RF_DEPTH,
            min_samples_leaf=RF_MIN_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_all[tr_m], y_all[tr_m])
        oof_rf[vl_m] = rf.predict_proba(X_all[vl_m])

    meta_train = np.hstack([oof_lr, oof_nb, oof_rf])
    meta_lr = LogisticRegression(
        max_iter=5000, C=META_C, solver="lbfgs", random_state=RANDOM_STATE
    )
    meta_lr.fit(meta_train, y_all)

    lr_f = LogisticRegression(
        max_iter=2500, C=LR_C, solver="lbfgs", random_state=RANDOM_STATE
    )
    lr_f.fit(X_all, y_all)

    nb_f = NaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
    cnb_f = CNaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
    nb_f.fit(X_nb_all, y_all)
    cnb_f.fit(X_nb_all, y_all)

    rf_f = RandomForestClassifier(
        n_estimators=RF_N_EST,
        max_depth=RF_DEPTH,
        min_samples_leaf=RF_MIN_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_f.fit(X_all, y_all)

    trees, max_n = _export_forest(rf_f)
    n_trees = len(trees)

    def _nb_save(m, prefix):
        d = {
            f"{prefix}_log_priors": m.log_priors_,
            f"{prefix}_log_likelihoods": m.log_likelihoods_,
            f"{prefix}_mn_idx": m._mn_idx.astype(np.int64),
            f"{prefix}_g_idx": m._g_idx.astype(np.int64),
        }
        if m.gauss_mean_ is not None:
            d[f"{prefix}_gauss_mean"] = m.gauss_mean_
            d[f"{prefix}_gauss_std"] = m.gauss_std_
        return d

    w = {
        "lr_coef": lr_f.coef_.astype(np.float64),
        "lr_intercept": lr_f.intercept_.astype(np.float64),
        "meta_coef": meta_lr.coef_.astype(np.float64),
        "meta_intercept": meta_lr.intercept_.astype(np.float64),
        "rf_n_trees": np.array([n_trees], dtype=np.int32),
        "rf_max_nodes": np.array([max_n], dtype=np.int32),
        "rf_active_nodes": np.array([t[5] for t in trees], dtype=np.int32),
    }
    w.update(_nb_save(nb_f, "nb"))
    w.update(_nb_save(cnb_f, "cnb"))

    for i, (cl, cr, feat, thr, val, nn) in enumerate(trees):
        w[f"rf{i}_cl"] = cl
        w[f"rf{i}_cr"] = cr
        w[f"rf{i}_feat"] = feat
        w[f"rf{i}_thr"] = thr
        w[f"rf{i}_val"] = val

    np.savez_compressed(os.path.join(out_dir, "model_weights.npz"), **w)

    def _jsonify(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(kk): _jsonify(vv) for kk, vv in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(x) for x in obj]
        return obj

    nb_json = _jsonify(nb_fit_info)

    bundle = {
        "fit_state": _serialize_state(state),
        "nb_fit_info": nb_json,
        "nb_feat_params": NB_FEAT_PARAMS,
        "nb_blend": NB_BLEND,
        "gaussian_cols": g_cols,
        "classes": classes,
        "class_to_idx": {str(k): int(v) for k, v in class_to_idx.items()},
    }
    with open(os.path.join(out_dir, "model_state.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print("Wrote model_state.json and model_weights.npz to", out_dir)


if __name__ == "__main__":
    main()
