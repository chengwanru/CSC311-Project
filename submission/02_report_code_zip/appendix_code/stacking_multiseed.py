# Allow imports from repository root when this file lives in appendix_code/.
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
Multi-seed ensemble stacking: K independent (LR + NB + RF) triplets → meta-LR.

Motivation
----------
Averaging K OOF runs compresses them into 9 numbers, hiding per-seed information.
Stacking keeps all K×9 probability columns as distinct meta-features so the
meta-LR can learn different weights per seed — equivalent to bagging at the
stacking level.  With K=5, variance from any single bad seed is diluted
structurally rather than numerically.

Pipeline
--------
Stage 1  Hyperparameter CV
           5-fold person-level CV on the full cleaned data (no held-out test in CV).
           Selects best C for LR, best (alpha, blend) for NB,
           best (n_estimators, max_depth) for RF.
           (This is an exploratory variant; the submitted final model A uses fixed
           hyperparameters in ../stacking_ensemble.py instead of this grid search.)

Stage 2  Multi-seed OOF generation  [on 80% trainval pool]
           For each seed s in BASE_SEEDS:
             Person-level 5-fold CV on trainval, with fold-shuffle seed = s,
             LR random_state = s, RF random_state = s.
             Yields an OOF block of shape (n_trainval, 3*K).
           Horizontally concatenate K blocks → meta_train shape (n_trainval, K*3*K).

Stage 3  Meta-LR training
           LogisticRegression (C=META_C) on the full K*3*K-dim OOF meta-features.

Stage 4  Final evaluation
           For each seed s, refit LR(s) + NB + RF(s) on full trainval.
           Collect K probability blocks of shape (n_test, 3*K), hstack →
           meta_test (n_test, K*3*K) → meta_lr.predict → final predictions.

Evaluation
----------
- Accuracy and macro-F1 for each base model (averaged over seeds) and stacking.
- Per-class confusion matrix for the stacking model.
- Per-seed base model accuracy breakdown.

BASE_SEEDS = [42, 7, 13, 99, 123]   # K = 5

Run from repo root: python appendix_code/stacking_multiseed.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from pipeline import (
    CSV_PATH, RANDOM_STATE,
    clean, fit_state, transform_df,
    regular_split, grouped_kfold_split,
)
from naive_bayes import (
    NaiveBayes,
    CNaiveBayes,
    build_features,
    apply_weights,
    gaussian_col_indices,
)

# ── Hyperparameter grids (Stage 1 CV) ────────────────────────────────────────
LR_C_GRID     = [10, 100]
NB_ALPHA_GRID = [0.5, 0.9, 1.5]
NB_BLEND_GRID = [0.8, 0.9, 1.0]
RF_NEST_GRID  = [100, 200]
RF_DEPTH_GRID = [None, 15]

K_FOLDS    = 5
META_C     = 1.0
BASE_SEEDS = [42, 7, 13, 99, 123]   # K = 5 independent model seeds

# ── Fixed pipeline params ────────────────────────────────────────────────────
MAX_FEATURES    = 6000
MIN_DF          = 1
LR_CLASS_WEIGHT = None

NB_FEAT_PARAMS = {
    "vocab_size": 2000, "alpha": 1.0, "binary_tf": True, "num_bins": None,
    "blend_weight": 1.0, "w_likert": 1.0, "w_numeric": 0.5,
    "w_price": 0.0, "w_categorical": 1.0,
}

SORT_KEY = ["unique_id", "Painting"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nb_proba(X, nb_m, cnb_m, blend):
    """Blended NB+CNB log-posteriors → softmax probabilities (n, K)."""
    nb_lp  = nb_m.predict_log_proba(X)
    cnb_lp = cnb_m.predict_log_proba(X)
    nb_lp  -= nb_lp.max(axis=1, keepdims=True)
    cnb_lp -= cnb_lp.max(axis=1, keepdims=True)
    blended = blend * nb_lp + (1.0 - blend) * cnb_lp
    e = np.exp(blended - blended.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _report(name, y_true, pred):
    acc = accuracy_score(y_true, pred)
    f1  = f1_score(y_true, pred, average="macro", zero_division=0)
    print(f"  {name:52s}  acc={acc:.4f}  macro_f1={f1:.4f}")
    return acc, f1


def _print_cm(y_true, y_pred, class_names):
    """Print a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cw = 16
    print("    " + f"{'Pred ->':>{cw}}" + "".join(f"{s:>{cw}}" for s in class_names))
    print("    " + "-" * cw * (len(class_names) + 1))
    for i, lbl in enumerate(class_names):
        print("    " + f"{lbl:>{cw - 2}}" +
              "".join(f"{cm[i, j]:>{cw}}" for j in range(len(class_names))))


def _oof_one_seed(
    seed, df_tv, X_tv, y_tv, X_nb_tv, y_nb_tv,
    best_lr_C, best_nb_alpha, best_nb_blend,
    best_rf_nest, best_rf_depth, g_cols, K,
):
    """One full 5-fold OOF pass with fold-shuffle seed = model seed = `seed`.

    Returns
    -------
    block : ndarray, shape (n_tv, 3*K)
        Horizontally stacked OOF probabilities [p_LR | p_NB | p_RF].
    oof_accs : tuple (acc_lr, acc_nb, acc_rf)
    """
    n_tv     = len(y_tv)
    block_lr = np.zeros((n_tv, K))
    block_nb = np.zeros((n_tv, K))
    block_rf = np.zeros((n_tv, K))

    rng      = np.random.RandomState(seed)
    tv_ids   = df_tv["unique_id"].unique().copy()
    rng.shuffle(tv_ids)
    id_folds = np.array_split(tv_ids, K_FOLDS)

    for fold_i, fold_val_ids in enumerate(id_folds):
        fold_val_set   = set(fold_val_ids)
        fold_train_set = set(np.concatenate(
            [id_folds[j] for j in range(K_FOLDS) if j != fold_i]
        ))
        tr_mask = df_tv["unique_id"].isin(fold_train_set).values
        vl_mask = df_tv["unique_id"].isin(fold_val_set).values

        lr_f = LogisticRegression(
            max_iter=2500, C=best_lr_C, class_weight=LR_CLASS_WEIGHT,
            solver="lbfgs", random_state=seed)
        lr_f.fit(X_tv[tr_mask], y_tv[tr_mask])
        block_lr[vl_mask] = lr_f.predict_proba(X_tv[vl_mask])

        nb_f  = NaiveBayes(alpha=best_nb_alpha,  gaussian_cols=g_cols)
        cnb_f = CNaiveBayes(alpha=best_nb_alpha, gaussian_cols=g_cols)
        nb_f.fit(X_nb_tv[tr_mask],  y_nb_tv[tr_mask])
        cnb_f.fit(X_nb_tv[tr_mask], y_nb_tv[tr_mask])
        block_nb[vl_mask] = _nb_proba(X_nb_tv[vl_mask], nb_f, cnb_f, best_nb_blend)

        rf_f = RandomForestClassifier(
            n_estimators=best_rf_nest, max_depth=best_rf_depth,
            random_state=seed, n_jobs=-1)
        rf_f.fit(X_tv[tr_mask], y_tv[tr_mask])
        block_rf[vl_mask] = rf_f.predict_proba(X_tv[vl_mask])

    block    = np.hstack([block_lr, block_nb, block_rf])
    oof_accs = (
        accuracy_score(y_tv, np.argmax(block_lr, axis=1)),
        accuracy_score(y_tv, np.argmax(block_nb, axis=1)),
        accuracy_score(y_tv, np.argmax(block_rf, axis=1)),
    )
    return block, oof_accs


def _refit_one_seed(
    seed, X_tv, y_tv, X_nb_tv, y_nb_tv, X_test, X_nb_test,
    best_lr_C, best_nb_alpha, best_nb_blend,
    best_rf_nest, best_rf_depth, g_cols,
):
    """Refit base models on full trainval with seed `seed`.

    Returns
    -------
    p_lr, p_nb, p_rf : each ndarray shape (n_test, K)
    """
    lr80 = LogisticRegression(
        max_iter=2500, C=best_lr_C, class_weight=LR_CLASS_WEIGHT,
        solver="lbfgs", random_state=seed)
    lr80.fit(X_tv, y_tv)
    p_lr = lr80.predict_proba(X_test)

    nb80  = NaiveBayes(alpha=best_nb_alpha,  gaussian_cols=g_cols)
    cnb80 = CNaiveBayes(alpha=best_nb_alpha, gaussian_cols=g_cols)
    nb80.fit(X_nb_tv,  y_nb_tv)
    cnb80.fit(X_nb_tv, y_nb_tv)
    p_nb = _nb_proba(X_nb_test, nb80, cnb80, best_nb_blend)

    rf80 = RandomForestClassifier(
        n_estimators=best_rf_nest, max_depth=best_rf_depth,
        random_state=seed, n_jobs=-1)
    rf80.fit(X_tv, y_tv)
    p_rf = rf80.predict_proba(X_test)

    return p_lr, p_nb, p_rf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    K_seeds  = len(BASE_SEEDS)
    meta_dim_per_seed = 9   # 3 models × 3 classes

    print(f"=== Multi-seed ensemble stacking ===")
    print(f"BASE_SEEDS = {BASE_SEEDS}")
    print(f"Meta-feature dim = {K_seeds} seeds × {meta_dim_per_seed} = {K_seeds * meta_dim_per_seed}")

    # ── Load, clean, split ────────────────────────────────────────────────
    df_clean = clean(pd.read_csv(CSV_PATH))

    train_df, val_df, test_df = regular_split(df_clean, random_state=RANDOM_STATE)
    train_df = train_df.sort_values(SORT_KEY).reset_index(drop=True)
    val_df   = val_df.sort_values(SORT_KEY).reset_index(drop=True)
    test_df  = test_df.sort_values(SORT_KEY).reset_index(drop=True)

    state = fit_state(train_df, max_features=MAX_FEATURES, min_df=MIN_DF)
    X_train, y_train = transform_df(train_df, state)
    X_val,   y_val   = transform_df(val_df,   state)
    X_test,  y_test  = transform_df(test_df,  state)

    class_to_idx = {c: i for i, c in enumerate(state["classes"])}
    classes = state["classes"]
    K       = len(classes)

    X_nb_train, nb_fit_info = build_features(train_df, fit_info=None,         params=NB_FEAT_PARAMS)
    X_nb_val,   _           = build_features(val_df,   fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_test,  _           = build_features(test_df,  fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_train = apply_weights(X_nb_train, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    X_nb_val   = apply_weights(X_nb_val,   NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    X_nb_test  = apply_weights(X_nb_test,  NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    g_cols = gaussian_col_indices(NB_FEAT_PARAMS)

    n_tv = len(train_df) + len(val_df)
    print(f"\nClasses                  : {classes}")
    print(f"Train / val / test rows  : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"LR/RF feature dim        : {X_train.shape[1]}")
    print(f"NB feature dim           : {X_nb_train.shape[1]}")

    # ── Stage 1: 5-fold CV hyperparameter selection ───────────────────────
    print(f"\n[Stage 1] 5-fold CV for model hyperparameter selection...")

    cv_fold_ids = [
        (set(tr["unique_id"]), set(vl["unique_id"]))
        for tr, vl, _ in grouped_kfold_split(df_clean, k=K_FOLDS,
                                              random_state=RANDOM_STATE)
    ]

    def _lr_cv(C):
        accs = []
        for tr_ids, vl_ids in cv_fold_ids:
            tr = df_clean[df_clean["unique_id"].isin(tr_ids)].sort_values(SORT_KEY).reset_index(drop=True)
            vl = df_clean[df_clean["unique_id"].isin(vl_ids)].sort_values(SORT_KEY).reset_index(drop=True)
            Xtr, ytr = transform_df(tr, state)
            Xvl, yvl = transform_df(vl, state)
            m = LogisticRegression(max_iter=2500, C=C, class_weight=LR_CLASS_WEIGHT,
                                   solver="lbfgs", random_state=RANDOM_STATE)
            m.fit(Xtr, ytr)
            accs.append(accuracy_score(yvl, m.predict(Xvl)))
        return float(np.mean(accs))

    def _nb_cv(alpha, blend):
        accs = []
        for tr_ids, vl_ids in cv_fold_ids:
            tr = df_clean[df_clean["unique_id"].isin(tr_ids)].sort_values(SORT_KEY).reset_index(drop=True)
            vl = df_clean[df_clean["unique_id"].isin(vl_ids)].sort_values(SORT_KEY).reset_index(drop=True)
            Xtr, _ = build_features(tr, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
            Xvl, _ = build_features(vl, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
            Xtr = apply_weights(Xtr, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
            Xvl = apply_weights(Xvl, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
            ytr = np.array([class_to_idx[c] for c in tr["Painting"]])
            yvl = np.array([class_to_idx[c] for c in vl["Painting"]])
            nb_m  = NaiveBayes(alpha=alpha,  gaussian_cols=g_cols)
            cnb_m = CNaiveBayes(alpha=alpha, gaussian_cols=g_cols)
            nb_m.fit(Xtr, ytr); cnb_m.fit(Xtr, ytr)
            accs.append(accuracy_score(yvl, np.argmax(_nb_proba(Xvl, nb_m, cnb_m, blend), axis=1)))
        return float(np.mean(accs))

    def _rf_cv(n_est, depth):
        accs = []
        for tr_ids, vl_ids in cv_fold_ids:
            tr = df_clean[df_clean["unique_id"].isin(tr_ids)].sort_values(SORT_KEY).reset_index(drop=True)
            vl = df_clean[df_clean["unique_id"].isin(vl_ids)].sort_values(SORT_KEY).reset_index(drop=True)
            Xtr, ytr = transform_df(tr, state)
            Xvl, yvl = transform_df(vl, state)
            m = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                       random_state=RANDOM_STATE, n_jobs=-1)
            m.fit(Xtr, ytr)
            accs.append(accuracy_score(yvl, m.predict(Xvl)))
        return float(np.mean(accs))

    lr_scores = {C: _lr_cv(C) for C in LR_C_GRID}
    best_lr_C = max(lr_scores, key=lr_scores.get)
    print(f"  LR  CV scores: { {C: f'{v:.4f}' for C, v in lr_scores.items()} }"
          f"  -> best C = {best_lr_C}")

    nb_scores       = {(a, b): _nb_cv(a, b) for a in NB_ALPHA_GRID for b in NB_BLEND_GRID}
    best_nb_key     = max(nb_scores, key=nb_scores.get)
    best_nb_alpha, best_nb_blend = best_nb_key
    print(f"  NB  best (alpha={best_nb_alpha}, blend={best_nb_blend})"
          f"  CV acc = {nb_scores[best_nb_key]:.4f}")

    rf_scores      = {(n, d): _rf_cv(n, d) for n in RF_NEST_GRID for d in RF_DEPTH_GRID}
    best_rf_key    = max(rf_scores, key=rf_scores.get)
    best_rf_nest, best_rf_depth = best_rf_key
    print(f"  RF  best (n_est={best_rf_nest}, depth={best_rf_depth})"
          f"  CV acc = {rf_scores[best_rf_key]:.4f}")

    print(f"\n  CV-selected: LR C={best_lr_C} | NB alpha={best_nb_alpha}"
          f" blend={best_nb_blend} | RF n_est={best_rf_nest} depth={best_rf_depth}")

    # ── Stage 2: Multi-seed OOF generation ───────────────────────────────
    df_tv   = pd.concat([train_df, val_df], ignore_index=True)
    X_tv    = np.vstack([X_train, X_val])
    y_tv    = np.concatenate([y_train, y_val])
    X_nb_tv = np.vstack([X_nb_train, X_nb_val])
    y_nb_tv = np.concatenate([
        np.array([class_to_idx[c] for c in train_df["Painting"]]),
        np.array([class_to_idx[c] for c in val_df["Painting"]]),
    ])

    print(f"\n[Stage 2] Multi-seed OOF on {n_tv}-row trainval"
          f" (K={K_seeds} seeds, each {K_FOLDS}-fold → {K_seeds * K_FOLDS} fold fits per model)...")

    oof_blocks = []
    for s in BASE_SEEDS:
        print(f"  Seed {s:>5d} ...", end="  ", flush=True)
        block, (acc_lr, acc_nb, acc_rf) = _oof_one_seed(
            s, df_tv, X_tv, y_tv, X_nb_tv, y_nb_tv,
            best_lr_C, best_nb_alpha, best_nb_blend,
            best_rf_nest, best_rf_depth, g_cols, K,
        )
        print(f"OOF acc  LR={acc_lr:.4f}  NB={acc_nb:.4f}  RF={acc_rf:.4f}")
        oof_blocks.append(block)

    meta_train = np.hstack(oof_blocks)   # (n_tv, K_seeds * 3 * K)
    meta_dim   = meta_train.shape[1]
    print(f"\n  Meta-feature matrix: {meta_train.shape}"
          f"  [{K_seeds} seeds × 3 models × {K} classes]")

    # ── Stage 3: Train meta-LR on K_seeds×9-dim OOF meta-features ────────
    print(f"\n[Stage 3] Training meta-LR (C={META_C:g}) on {meta_dim}-dim OOF features...")
    meta_lr = LogisticRegression(
        max_iter=5000, C=META_C, solver="lbfgs", random_state=RANDOM_STATE)
    meta_lr.fit(meta_train, y_tv)

    # Summarise learned weights as L1 norm per [seed × model] block.
    # Each block spans 3 columns (K=3 class probabilities).
    coef = meta_lr.coef_   # shape (K, meta_dim)
    print(f"\n  Meta-LR coeff L1-norm per [seed × model] block "
          f"(each row = one predicted class):")
    header = f"  {'':14s}"
    for s in BASE_SEEDS:
        header += f"  {'s'+str(s)+':LR':>9}  {'s'+str(s)+':NB':>9}  {'s'+str(s)+':RF':>9}"
    print(header)
    for ci, cname in enumerate(classes):
        row = f"  {cname:14s}"
        for si in range(K_seeds):
            base = si * 3 * K
            row += (f"  {np.abs(coef[ci, base      : base+K  ]).sum():>9.3f}"
                    f"  {np.abs(coef[ci, base+K    : base+2*K]).sum():>9.3f}"
                    f"  {np.abs(coef[ci, base+2*K  : base+3*K]).sum():>9.3f}")
        print(row)

    # ── Stage 4: Refit per seed, evaluate on test ─────────────────────────
    print(f"\n[Stage 4] Refitting {K_seeds} × 3 base models on full {n_tv}-row trainval,"
          f" evaluating on {len(y_test)}-row test set...")

    test_blocks  = []
    p_lr_sum     = np.zeros((len(y_test), K))
    p_nb_sum     = np.zeros((len(y_test), K))
    p_rf_sum     = np.zeros((len(y_test), K))

    for s in BASE_SEEDS:
        p_lr, p_nb, p_rf = _refit_one_seed(
            s, X_tv, y_tv, X_nb_tv, y_nb_tv, X_test, X_nb_test,
            best_lr_C, best_nb_alpha, best_nb_blend,
            best_rf_nest, best_rf_depth, g_cols,
        )
        test_blocks.append(np.hstack([p_lr, p_nb, p_rf]))
        p_lr_sum += p_lr
        p_nb_sum += p_nb
        p_rf_sum += p_rf

    meta_test  = np.hstack(test_blocks)   # (n_test, meta_dim)
    pred_stack = meta_lr.predict(meta_test)

    # Averaged base-model predictions for comparison
    pred_lr_avg = np.argmax(p_lr_sum, axis=1)
    pred_nb_avg = np.argmax(p_nb_sum, axis=1)
    pred_rf_avg = np.argmax(p_rf_sum, axis=1)

    print(f"\n{'─'*72}")
    print(f"  TEST RESULTS  (n = {len(y_test)})")
    print(f"{'─'*72}")
    _report(f"LR  (avg over {K_seeds} seeds)",    y_test, pred_lr_avg)
    _report(f"NB  (avg over {K_seeds} seeds)",    y_test, pred_nb_avg)
    _report(f"RF  (avg over {K_seeds} seeds)",    y_test, pred_rf_avg)
    print()
    _report(f"Multi-seed stacking  (meta-LR, {meta_dim}-dim)", y_test, pred_stack)
    print(f"{'─'*72}")

    print(f"\n  Confusion matrix — Multi-seed stacking (meta-LR, {meta_dim}-dim):")
    _print_cm(y_test, pred_stack, classes)

    # ── Per-seed base-model breakdown ─────────────────────────────────────
    print(f"\n  Per-seed base-model accuracy on test set:")
    print(f"  {'seed':>6}  {'LR':>8}  {'NB':>8}  {'RF':>8}")
    for si, s in enumerate(BASE_SEEDS):
        blk = test_blocks[si]
        a_lr = accuracy_score(y_test, np.argmax(blk[:,      :K  ], axis=1))
        a_nb = accuracy_score(y_test, np.argmax(blk[:, K    :2*K], axis=1))
        a_rf = accuracy_score(y_test, np.argmax(blk[:, 2*K  :   ], axis=1))
        print(f"  {s:>6d}  {a_lr:>8.4f}  {a_nb:>8.4f}  {a_rf:>8.4f}")

    stk_acc = accuracy_score(y_test, pred_stack)
    stk_f1  = f1_score(y_test, pred_stack, average="macro", zero_division=0)
    print(f"\n  Final stacking accuracy : {stk_acc:.4f}")
    print(f"  Final stacking macro-F1 : {stk_f1:.4f}")


if __name__ == "__main__":
    main()
