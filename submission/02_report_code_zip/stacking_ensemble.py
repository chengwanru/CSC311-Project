"""
Stacking ensemble: LR + NB (custom blend) + RF → meta logistic regression.

Final configuration (multiseed quick evaluation, min≈0.923 / mean≈0.938 on 6 seeds):
  LR:  C=100 (fixed, no CV)
  NB:  alpha=0.9, NB/CNB blend=1.0
  RF:  n_estimators=200, max_depth=None, min_samples_leaf=1
  Meta: LogisticRegression C=0.5 on 9-dim OOF base probabilities

Data: training_data.csv via pipeline.clean(). Person-level 60/20/20 split;
      stacking OOF + refit on train+val (80%). Rows sorted by (unique_id, Painting).

For MarkUs prediction (no sklearn in pred.py): train on ALL data and export with
  python export_model.py
then submit pred.py + model_state.json + model_weights.npz.

Run: python stacking_ensemble.py

Programmatic use (e.g. report figures):
  from stacking_ensemble import run_stacking_eval
  r = run_stacking_eval(split_seed=42)
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from pipeline import CSV_PATH, RANDOM_STATE, clean, fit_state, transform_df, regular_split
from naive_bayes import (
    NaiveBayes,
    CNaiveBayes,
    build_features,
    apply_weights,
    gaussian_col_indices,
)

K_FOLDS = 5
META_C = 0.5
LR_C = 100
NB_ALPHA = 0.9
NB_BLEND = 1.0
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1

MAX_FEATURES = 6000
MIN_DF = 1
LR_CLASS_WEIGHT = None

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


def _majority_vote_3(plr, pnb, prf):
    stack = np.stack([plr, pnb, prf], axis=1)
    out = np.empty(len(stack), dtype=int)
    for i, row in enumerate(stack):
        vals, counts = np.unique(row, return_counts=True)
        if counts.max() >= 2:
            out[i] = int(vals[np.argmax(counts)])
        else:
            out[i] = int(row[0])
    return out


def _print_cm(y_true, y_pred):
    short = ["Persistence", "Starry Night", "Water Lily"]
    cm = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            cm[i, j] = int(np.sum((y_true == i) & (y_pred == j)))
    cw = 14
    print("    " + f"{'Pred ->':>{cw}}" + "".join(f"{s:>{cw}}" for s in short))
    print("    " + "-" * cw * 4)
    for i, lbl in enumerate(short):
        print("    " + f"{lbl:>{cw-2}}" + "".join(f"{cm[i,j]:>{cw}}" for j in range(3)))


def _report(name, y_true, pred):
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    print(f"  {name:44s}  acc={acc:.4f}  macro_f1={f1:.4f}")
    return acc, f1


def run_stacking_eval(
    split_seed: Optional[int] = None, *, verbose: bool = True
) -> Dict[str, Any]:
    """
    Train/evaluate the final stacking pipeline (model A) with a person-level split.

    Parameters
    ----------
    split_seed
        Seed for ``regular_split`` and for sklearn ``random_state`` in bases/meta.
        Default: ``pipeline.RANDOM_STATE`` (42).
    verbose
        If False, suppress printed progress (for batch runs, e.g. ``report_figures.py``).

    Returns
    -------
    dict with keys including ``y_test``, ``pred_stack``, ``pred_lr``, ``pred_nb``,
    ``pred_rf``, ``pred_maj``, per-model accuracies, ``macro_f1_stack``,
    ``confusion_matrix`` (3x3 int, true rows / pred cols), ``class_names``,
    ``split_seed``, ``n_train``, ``n_val``, ``n_test``.
    """
    seed = RANDOM_STATE if split_seed is None else int(split_seed)
    out_cm = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with out_cm:
        return _run_stacking_eval_core(seed)


def _run_stacking_eval_core(seed: int) -> Dict[str, Any]:
    df_clean = clean(pd.read_csv(CSV_PATH))

    train_df, val_df, test_df = regular_split(df_clean, random_state=seed)
    train_df = train_df.sort_values(SORT_KEY).reset_index(drop=True)
    val_df = val_df.sort_values(SORT_KEY).reset_index(drop=True)
    test_df = test_df.sort_values(SORT_KEY).reset_index(drop=True)

    state = fit_state(train_df, max_features=MAX_FEATURES, min_df=MIN_DF)
    X_train, y_train = transform_df(train_df, state)
    X_val, y_val = transform_df(val_df, state)
    X_test, y_test = transform_df(test_df, state)

    class_to_idx = {c: i for i, c in enumerate(state["classes"])}

    X_nb_train, nb_fit_info = build_features(train_df, fit_info=None, params=NB_FEAT_PARAMS)
    X_nb_val, _ = build_features(val_df, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_test, _ = build_features(test_df, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_train = apply_weights(X_nb_train, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    X_nb_val = apply_weights(X_nb_val, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    X_nb_test = apply_weights(X_nb_test, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    g_cols = gaussian_col_indices(NB_FEAT_PARAMS)

    classes = state["classes"]
    K = len(classes)
    n_tv = len(train_df) + len(val_df)
    print("Classes:", classes)
    print(f"Train / val / test rows  : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"LR/RF feature dim        : {X_train.shape[1]}")
    print(f"NB feature dim           : {X_nb_train.shape[1]}")
    print(
        f"\n[Fixed hyperparameters] LR C={LR_C} | NB alpha={NB_ALPHA} blend={NB_BLEND} | "
        f"RF n_est={RF_N_ESTIMATORS} depth={RF_MAX_DEPTH} min_leaf={RF_MIN_SAMPLES_LEAF} | "
        f"meta C={META_C}"
    )

    df_tv = pd.concat([train_df, val_df], ignore_index=True)
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    X_nb_tv = np.vstack([X_nb_train, X_nb_val])
    y_nb_tv = np.concatenate(
        [
            np.array([class_to_idx[c] for c in train_df["Painting"]]),
            np.array([class_to_idx[c] for c in val_df["Painting"]]),
        ]
    )

    print(f"\n[Stage 2] OOF prediction generation on {n_tv}-row trainval pool...")

    rng = np.random.RandomState(seed)
    tv_ids = df_tv["unique_id"].unique()
    rng.shuffle(tv_ids)
    id_folds = np.array_split(tv_ids, K_FOLDS)

    oof_lr = np.zeros((n_tv, K))
    oof_nb = np.zeros((n_tv, K))
    oof_rf = np.zeros((n_tv, K))

    for fold_i, fold_val_ids in enumerate(id_folds):
        fold_val_set = set(fold_val_ids)
        fold_train_set = set(
            np.concatenate([id_folds[j] for j in range(K_FOLDS) if j != fold_i])
        )

        tr_mask = df_tv["unique_id"].isin(fold_train_set).values
        vl_mask = df_tv["unique_id"].isin(fold_val_set).values

        print(f"  Fold {fold_i+1}/{K_FOLDS}: train={tr_mask.sum()}  val={vl_mask.sum()}")

        lr_f = LogisticRegression(
            max_iter=2500,
            C=LR_C,
            class_weight=LR_CLASS_WEIGHT,
            solver="lbfgs",
            random_state=seed,
        )
        lr_f.fit(X_tv[tr_mask], y_tv[tr_mask])
        oof_lr[vl_mask] = lr_f.predict_proba(X_tv[vl_mask])

        nb_f = NaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
        cnb_f = CNaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
        nb_f.fit(X_nb_tv[tr_mask], y_nb_tv[tr_mask])
        cnb_f.fit(X_nb_tv[tr_mask], y_nb_tv[tr_mask])
        oof_nb[vl_mask] = _nb_proba(X_nb_tv[vl_mask], nb_f, cnb_f, NB_BLEND)

        rf_f = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=seed,
            n_jobs=-1,
        )
        rf_f.fit(X_tv[tr_mask], y_tv[tr_mask])
        oof_rf[vl_mask] = rf_f.predict_proba(X_tv[vl_mask])

    print(f"\n--- OOF base model performance ({n_tv} trainval rows) ---")
    _report("LR  (OOF)", y_tv, np.argmax(oof_lr, axis=1))
    _report("NB  (OOF)", y_tv, np.argmax(oof_nb, axis=1))
    _report("RF  (OOF)", y_tv, np.argmax(oof_rf, axis=1))

    meta_train = np.hstack([oof_lr, oof_nb, oof_rf])
    print(f"\n  Meta-feature matrix: {meta_train.shape}")

    print(f"\n[Stage 3] Training meta-LR (C={META_C}) on OOF features...")
    meta_lr = LogisticRegression(
        max_iter=5000, C=META_C, solver="lbfgs", random_state=seed
    )
    meta_lr.fit(meta_train, y_tv)

    coef = meta_lr.coef_
    feat_names = [f"LR_c{k}" for k in range(K)] + [f"NB_c{k}" for k in range(K)] + [f"RF_c{k}" for k in range(K)]
    short_cls = ["Persistence", "StarryNight", "WaterLily"]
    print("\n  Meta-LR coefficients:")
    print(f"  {'':18s}" + "".join(f"{n:>10s}" for n in feat_names))
    for i, cname in enumerate(short_cls):
        print(f"  {cname:18s}" + "".join(f"{coef[i, j]:>10.3f}" for j in range(9)))

    print(f"\n[Stage 4] Refit bases on 80%, evaluate on held-out 20% test...")

    lr80 = LogisticRegression(
        max_iter=2500,
        C=LR_C,
        class_weight=LR_CLASS_WEIGHT,
        solver="lbfgs",
        random_state=seed,
    )
    lr80.fit(X_tv, y_tv)
    p_lr_test = lr80.predict_proba(X_test)

    nb80 = NaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
    cnb80 = CNaiveBayes(alpha=NB_ALPHA, gaussian_cols=g_cols)
    nb80.fit(X_nb_tv, y_nb_tv)
    cnb80.fit(X_nb_tv, y_nb_tv)
    p_nb_test = _nb_proba(X_nb_test, nb80, cnb80, NB_BLEND)

    rf80 = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=-1,
    )
    rf80.fit(X_tv, y_tv)
    p_rf_test = rf80.predict_proba(X_test)

    meta_test = np.hstack([p_lr_test, p_nb_test, p_rf_test])
    pred_stack = meta_lr.predict(meta_test)

    pred_lr = np.argmax(p_lr_test, axis=1)
    pred_nb = np.argmax(p_nb_test, axis=1)
    pred_rf = np.argmax(p_rf_test, axis=1)
    pred_maj = _majority_vote_3(pred_lr, pred_nb, pred_rf)

    print(f"\n--- Test results (n={len(y_test)}) ---")
    _report("LR  (base model)", y_test, pred_lr)
    _report("NB  (base model)", y_test, pred_nb)
    _report("RF  (base model)", y_test, pred_rf)
    _report("Majority vote (3 models)", y_test, pred_maj)
    print()
    _report("Stacking  (meta-LR, 9-dim)", y_test, pred_stack)

    print(f"\n  Confusion matrix — Stacking:")
    _print_cm(y_test, pred_stack)

    agree = (pred_stack == pred_maj).sum()
    print(f"\n  Stacking agrees with majority vote on {agree}/{len(y_test)}"
          f" ({100*agree/len(y_test):.1f}%) of test rows.")

    cm = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            cm[i, j] = int(np.sum((y_test == i) & (pred_stack == j)))

    acc_lr, f1_lr = accuracy_score(y_test, pred_lr), f1_score(
        y_test, pred_lr, average="macro", zero_division=0
    )
    acc_nb, f1_nb = accuracy_score(y_test, pred_nb), f1_score(
        y_test, pred_nb, average="macro", zero_division=0
    )
    acc_rf, f1_rf = accuracy_score(y_test, pred_rf), f1_score(
        y_test, pred_rf, average="macro", zero_division=0
    )
    acc_maj, f1_maj = accuracy_score(y_test, pred_maj), f1_score(
        y_test, pred_maj, average="macro", zero_division=0
    )
    acc_stack, f1_stack = accuracy_score(y_test, pred_stack), f1_score(
        y_test, pred_stack, average="macro", zero_division=0
    )

    return {
        "split_seed": seed,
        "class_names": list(classes),
        "y_test": y_test,
        "pred_stack": pred_stack,
        "pred_lr": pred_lr,
        "pred_nb": pred_nb,
        "pred_rf": pred_rf,
        "pred_maj": pred_maj,
        "acc_lr": float(acc_lr),
        "acc_nb": float(acc_nb),
        "acc_rf": float(acc_rf),
        "acc_maj": float(acc_maj),
        "acc_stack": float(acc_stack),
        "macro_f1_lr": float(f1_lr),
        "macro_f1_nb": float(f1_nb),
        "macro_f1_rf": float(f1_rf),
        "macro_f1_maj": float(f1_maj),
        "macro_f1_stack": float(f1_stack),
        "confusion_matrix": cm,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }


def main() -> None:
    run_stacking_eval(split_seed=None, verbose=True)


if __name__ == "__main__":
    main()
