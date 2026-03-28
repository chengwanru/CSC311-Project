"""
Fast multi-seed check for 80/20 stacking (fixed LR/NB/RF + meta C).

Run: python stacking_seed_eval_fast.py
"""
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
META_C = 1.0

MAX_FEATURES = 6000
MIN_DF = 1
LR_CLASS_WEIGHT = None

LR_C = 100
NB_ALPHA = 0.9
NB_BLEND = 1.0
RF_N_EST = 400
RF_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 4

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


def _score(y, pred):
    return float(accuracy_score(y, pred)), float(
        f1_score(y, pred, average="macro", zero_division=0)
    )


def _fit_base(X, y, X_nb, y_nb, g_cols, seed):
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
        n_estimators=RF_N_EST,
        max_depth=RF_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return lr, nb, cnb, rf


def _oof_meta(df_fit, state, class_to_idx, nb_fit_info, g_cols, seed):
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
        )
        oof_lr[vl_mask] = lr.predict_proba(X_fit[vl_mask])
        oof_nb[vl_mask] = _nb_proba(X_nb_fit[vl_mask], nb, cnb, NB_BLEND)
        oof_rf[vl_mask] = rf.predict_proba(X_fit[vl_mask])

    meta_train = np.hstack([oof_lr, oof_nb, oof_rf])
    return X_fit, y_fit, X_nb_fit, y_nb_fit, meta_train


def eval_protocol(train_df, val_df, test_df, seed, use_trainval):
    train_df = train_df.sort_values(SORT_KEY).reset_index(drop=True)
    val_df = val_df.sort_values(SORT_KEY).reset_index(drop=True)
    test_df = test_df.sort_values(SORT_KEY).reset_index(drop=True)

    state = fit_state(train_df, max_features=MAX_FEATURES, min_df=MIN_DF)
    class_to_idx = {c: i for i, c in enumerate(state["classes"])}

    nb_fit_df = train_df if not use_trainval else pd.concat([train_df, val_df], ignore_index=True)
    _, nb_fit_info = build_features(nb_fit_df, fit_info=None, params=NB_FEAT_PARAMS)
    g_cols = gaussian_col_indices(NB_FEAT_PARAMS)

    fit_df = pd.concat([train_df, val_df], ignore_index=True) if use_trainval else train_df.copy()
    fit_df = fit_df.sort_values(SORT_KEY).reset_index(drop=True)

    X_fit, y_fit, X_nb_fit, y_nb_fit, meta_train = _oof_meta(
        fit_df, state, class_to_idx, nb_fit_info, g_cols, seed
    )
    meta_lr = LogisticRegression(
        max_iter=5000, C=META_C, solver="lbfgs", random_state=seed
    )
    meta_lr.fit(meta_train, y_fit)

    lr, nb, cnb, rf = _fit_base(X_fit, y_fit, X_nb_fit, y_nb_fit, g_cols, seed)
    X_test, y_test = transform_df(test_df, state)
    X_nb_test, _ = build_features(test_df, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_test = apply_weights(X_nb_test, NB_FEAT_PARAMS, nb_fit_info["block_cols"])

    p_lr = lr.predict_proba(X_test)
    p_nb = _nb_proba(X_nb_test, nb, cnb, NB_BLEND)
    p_rf = rf.predict_proba(X_test)
    meta_test = np.hstack([p_lr, p_nb, p_rf])
    pred = meta_lr.predict(meta_test)
    return _score(y_test, pred)


def main():
    df = clean(pd.read_csv(CSV_PATH))
    seeds = [1, 7, 13, 21, 42, 84]
    print("Seeds:", seeds)
    rows = []
    for sd in seeds:
        tr, va, te = regular_split(df, random_state=sd)
        acc60, f160 = eval_protocol(tr, va, te, sd, use_trainval=False)
        acc80, f180 = eval_protocol(tr, va, te, sd, use_trainval=True)
        rows.append((sd, acc60, acc80))
        print(
            f"seed={sd:>2d} | stack60 acc={acc60:.4f} f1={f160:.4f} || "
            f"stack80 acc={acc80:.4f} f1={f180:.4f}"
        )

    a60 = np.array([r[1] for r in rows], dtype=float)
    a80 = np.array([r[2] for r in rows], dtype=float)
    print("\nSummary (edit META_C / RF_* at top to match experiments)")
    print(
        f"60/20/20 acc mean±std: {a60.mean():.4f} ± {a60.std():.4f} | "
        f"min={a60.min():.4f} max={a60.max():.4f}"
    )
    print(
        f"80/20    acc mean±std: {a80.mean():.4f} ± {a80.std():.4f} | "
        f"min={a80.min():.4f} max={a80.max():.4f}"
    )


if __name__ == "__main__":
    main()
