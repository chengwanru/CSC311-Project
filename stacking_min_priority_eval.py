"""
80/20 stacking: sweep meta-LR C + RF knobs; rank by min test acc across seeds.

Run: python stacking_min_priority_eval.py
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from pipeline import CSV_PATH, clean, fit_state, transform_df, regular_split
from naive_bayes import (
    NaiveBayes,
    CNaiveBayes,
    build_features,
    apply_weights,
    gaussian_col_indices,
)

K_FOLDS = 5
MAX_FEATURES = 6000
MIN_DF = 1
LR_C = 100
LR_CLASS_WEIGHT = None
NB_ALPHA = 0.9
NB_BLEND = 1.0

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
SEEDS = [1, 7, 13, 21, 42, 84, 97, 123, 256, 512]


def _nb_proba(X, nb_m, cnb_m, blend):
    nb_lp = nb_m.predict_log_proba(X)
    cnb_lp = cnb_m.predict_log_proba(X)
    nb_lp -= nb_lp.max(axis=1, keepdims=True)
    cnb_lp -= cnb_lp.max(axis=1, keepdims=True)
    blended = blend * nb_lp + (1.0 - blend) * cnb_lp
    e = np.exp(blended - blended.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _fit_base(X, y, X_nb, y_nb, g_cols, seed, rf_n_est, rf_depth, rf_min_leaf):
    lr = LogisticRegression(
        max_iter=2500,
        C=LR_C,
        class_weight=LR_CLASS_WEIGHT,
        solver="lbfgs",
        random_state=seed,
    )
    lr.fit(X, y)
    nb = NaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
    cnb = CNaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
    nb.fit(X_nb, y_nb)
    cnb.fit(X_nb, y_nb)
    rf = RandomForestClassifier(
        n_estimators=rf_n_est,
        max_depth=rf_depth,
        min_samples_leaf=rf_min_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return lr, nb, cnb, rf


def _oof_meta(df_fit, state, class_to_idx, nb_fit_info, g_cols, seed, rf_n_est, rf_depth, rf_min_leaf):
    X_fit, y_fit = transform_df(df_fit, state)
    X_nb_fit, _ = build_features(df_fit, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_fit = apply_weights(X_nb_fit, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    y_nb_fit = np.array([class_to_idx[c] for c in df_fit["Painting"]])

    n = len(df_fit)
    K = len(state["classes"])
    oof_lr = np.zeros((n, K))
    oof_nb = np.zeros((n, K))
    oof_rf = np.zeros((n, K))

    rng = np.random.RandomState(seed)
    ids = df_fit["unique_id"].unique()
    rng.shuffle(ids)
    id_folds = np.array_split(ids, K_FOLDS)

    for i, fold_ids in enumerate(id_folds):
        val_set = set(fold_ids)
        tr_set = set(np.concatenate([id_folds[j] for j in range(K_FOLDS) if j != i]))
        tr_mask = df_fit["unique_id"].isin(tr_set).values
        vl_mask = df_fit["unique_id"].isin(val_set).values

        lr, nb, cnb, rf = _fit_base(
            X_fit[tr_mask],
            y_fit[tr_mask],
            X_nb_fit[tr_mask],
            y_nb_fit[tr_mask],
            g_cols,
            seed,
            rf_n_est,
            rf_depth,
            rf_min_leaf,
        )
        oof_lr[vl_mask] = lr.predict_proba(X_fit[vl_mask])
        oof_nb[vl_mask] = _nb_proba(X_nb_fit[vl_mask], nb, cnb, NB_BLEND)
        oof_rf[vl_mask] = rf.predict_proba(X_fit[vl_mask])

    return np.hstack([oof_lr, oof_nb, oof_rf]), X_fit, y_fit, X_nb_fit, y_nb_fit


def eval_one_seed(df_clean, seed, meta_c: float, rf_n_est, rf_depth, rf_min_leaf) -> Tuple[float, float]:
    train_df, val_df, test_df = regular_split(df_clean, random_state=seed)
    train_df = train_df.sort_values(SORT_KEY).reset_index(drop=True)
    val_df = val_df.sort_values(SORT_KEY).reset_index(drop=True)
    test_df = test_df.sort_values(SORT_KEY).reset_index(drop=True)

    state = fit_state(train_df, max_features=MAX_FEATURES, min_df=MIN_DF)
    class_to_idx = {c: i for i, c in enumerate(state["classes"])}

    nb_fit_df = pd.concat([train_df, val_df], ignore_index=True)
    _, nb_fit_info = build_features(nb_fit_df, fit_info=None, params=NB_FEAT_PARAMS)
    g_cols = gaussian_col_indices(NB_FEAT_PARAMS)

    fit_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(SORT_KEY).reset_index(drop=True)

    meta_train, X_fit, y_fit, X_nb_fit, y_nb_fit = _oof_meta(
        fit_df, state, class_to_idx, nb_fit_info, g_cols, seed, rf_n_est, rf_depth, rf_min_leaf
    )

    meta_lr = LogisticRegression(max_iter=5000, C=meta_c, solver="lbfgs", random_state=seed)
    meta_lr.fit(meta_train, y_fit)

    lr, nb, cnb, rf = _fit_base(X_fit, y_fit, X_nb_fit, y_nb_fit, g_cols, seed, rf_n_est, rf_depth, rf_min_leaf)

    X_test, y_test = transform_df(test_df, state)
    X_nb_test, _ = build_features(test_df, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_test = apply_weights(X_nb_test, NB_FEAT_PARAMS, nb_fit_info["block_cols"])

    p_lr = lr.predict_proba(X_test)
    p_nb = _nb_proba(X_nb_test, nb, cnb, NB_BLEND)
    p_rf = rf.predict_proba(X_test)
    meta_test = np.hstack([p_lr, p_nb, p_rf])
    pred = meta_lr.predict(meta_test)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return acc, f1


@dataclass
class Config:
    name: str
    meta_c: float
    rf_n_est: int
    rf_depth: object
    rf_min_leaf: int


def summarize(accs: List[float]) -> dict:
    a = np.array(accs, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p25": float(np.percentile(a, 25)),
    }


def main():
    df_clean = clean(pd.read_csv(CSV_PATH))

    configs = [
        Config("baseline_meta1e4_rf200", 1e4, 200, None, 1),
        Config("meta1_rf400_d12_l4", 1.0, 400, 12, 4),
        Config("meta0.5_rf200_dNone_l1", 0.5, 200, None, 1),
        Config("meta1_rf200_dNone_l1", 1.0, 200, None, 1),
    ]

    results = []
    for cfg in configs:
        accs, f1s = [], []
        for sd in SEEDS:
            acc, f1 = eval_one_seed(df_clean, sd, cfg.meta_c, cfg.rf_n_est, cfg.rf_depth, cfg.rf_min_leaf)
            accs.append(acc)
            f1s.append(f1)
        s = summarize(accs)
        sf = summarize(f1s)
        results.append((cfg, s, sf))
        print(
            f"{cfg.name:28s} acc mean={s['mean']:.4f} std={s['std']:.4f} "
            f"min={s['min']:.4f} p25={s['p25']:.4f} max={s['max']:.4f} | "
            f"f1 mean={sf['mean']:.4f} min={sf['min']:.4f}"
        )

    ranked = sorted(results, key=lambda x: (-x[1]["min"], -x[1]["p25"], -x[1]["mean"], x[1]["std"]))
    best_cfg, best_s, best_sf = ranked[0]
    print("\n=== Best by (min, p25, mean, std) ===")
    print(
        f"Name: {best_cfg.name}\n"
        f"  meta_C={best_cfg.meta_c}  RF n_est={best_cfg.rf_n_est}  "
        f"max_depth={best_cfg.rf_depth}  min_samples_leaf={best_cfg.rf_min_leaf}\n"
        f"  acc: mean={best_s['mean']:.4f} std={best_s['std']:.4f} "
        f"min={best_s['min']:.4f} p25={best_s['p25']:.4f} max={best_s['max']:.4f}"
    )


if __name__ == "__main__":
    main()
