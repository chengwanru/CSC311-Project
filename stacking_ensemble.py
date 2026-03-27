"""
Stacking ensemble: LR + NB (custom blend) + RF → meta logistic regression.

Motivation
----------
Rule-based gating (e.g. "trust NB when confident") is a hand-crafted decision.
Stacking replaces that heuristic with a learned meta-classifier: a logistic
regression trained on the out-of-fold (OOF) probability outputs of all three
base models.  The meta-classifier discovers automatically which model to trust
for which region of the input space.

Data
----
All three models share a single data source: training_data.csv, loaded and
cleaned once via pipeline.clean().  NB's build_features() parses Likert and
numeric columns inline (extract_likert / extract_numeric) so no separate
preprocessed.csv is needed.  All splits are person-level; within each split
rows are sorted by (unique_id, Painting) — LR/RF and NB feature arrays are
therefore always in identical row order with no alignment bookkeeping required.

Pipeline
--------
Stage 1  Model hyperparameter selection
           5-fold CV on df_clean — never sees the test set.
           Selects best C for LR, best (alpha, blend) for NB,
           best (n_estimators, max_depth) for RF.

Stage 2  Out-of-fold (OOF) prediction generation  [on 80% trainval pool]
           Person-level 5-fold CV on the trainval split.
           For each fold k, all three base models are trained on the other
           4 folds and their row-level probabilities are recorded on fold k.
           Meta-feature matrix shape: (n_trainval_rows, 9).

Stage 3  Meta-classifier training
           LogisticRegression (C=1e4, very light regularisation) trained on
           the 9-dim OOF meta-features.

Stage 4 / Protocol B  Final evaluation
           Refit all three base models on full 80% trainval.
           Generate test probabilities → 9-dim meta-features → meta-LR.

Hyperparameter grids (Stage 1):
  LR  : C               in {10, 100}
  NB  : alpha           in {0.5, 0.9, 1.5}
        blend_weight    in {0.8, 0.9, 1.0}
  RF  : n_estimators    in {100, 200}
        max_depth       in {None, 15}

Run: python stacking_ensemble.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

# ── Hyperparameter grids (Stage 1 CV) ─────────────────────────────────────────
LR_C_GRID     = [10, 100]
NB_ALPHA_GRID = [0.5, 0.9, 1.5]
NB_BLEND_GRID = [0.8, 0.9, 1.0]
RF_NEST_GRID  = [100, 200]
RF_DEPTH_GRID = [None, 15]

K_FOLDS  = 5
META_C   = 1e4   # very light regularisation for the meta-LR

# ── Fixed pipeline params ─────────────────────────────────────────────────────
MAX_FEATURES    = 6000
MIN_DF          = 1
LR_CLASS_WEIGHT = None

NB_FEAT_PARAMS = {
    "vocab_size": 2000, "alpha": 1.0, "binary_tf": True, "num_bins": None,
    "blend_weight": 1.0, "w_likert": 1.0, "w_numeric": 0.5,
    "w_price": 0.0, "w_categorical": 1.0,
}

# Sort key used throughout — guarantees identical row order across both datasets
SORT_KEY = ["unique_id", "Painting"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nb_proba(X, nb_m, cnb_m, blend):
    """Blended NB+CNB log-posteriors → softmax probabilities (n, K)."""
    nb_lp  = nb_m.predict_log_proba(X)
    cnb_lp = cnb_m.predict_log_proba(X)
    nb_lp  -= nb_lp.max(axis=1, keepdims=True)
    cnb_lp -= cnb_lp.max(axis=1, keepdims=True)
    blended = blend * nb_lp + (1.0 - blend) * cnb_lp
    e = np.exp(blended - blended.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _majority_vote_3(plr, pnb, prf):
    stack = np.stack([plr, pnb, prf], axis=1)
    out   = np.empty(len(stack), dtype=int)
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
    f1  = f1_score(y_true, pred, average="macro", zero_division=0)
    print(f"  {name:44s}  acc={acc:.4f}  macro_f1={f1:.4f}")
    return acc, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load, clean, and split ────────────────────────────────────────────────
    # All three models (LR, NB, RF) share the same cleaned DataFrame.
    df_clean = clean(pd.read_csv(CSV_PATH))

    train_df, val_df, test_df = regular_split(df_clean, random_state=RANDOM_STATE)
    train_df = train_df.sort_values(SORT_KEY).reset_index(drop=True)
    val_df   = val_df.sort_values(SORT_KEY).reset_index(drop=True)
    test_df  = test_df.sort_values(SORT_KEY).reset_index(drop=True)

    # LR/RF features
    state = fit_state(train_df, max_features=MAX_FEATURES, min_df=MIN_DF)
    X_train, y_train = transform_df(train_df, state)
    X_val,   y_val   = transform_df(val_df,   state)
    X_test,  y_test  = transform_df(test_df,  state)

    # NB features — same DataFrames as LR/RF (vocab fitted on train only)
    class_to_idx = {c: i for i, c in enumerate(state["classes"])}

    X_nb_train, nb_fit_info = build_features(train_df, fit_info=None,       params=NB_FEAT_PARAMS)
    X_nb_val,   _           = build_features(val_df,   fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_test,  _           = build_features(test_df,  fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_train = apply_weights(X_nb_train, NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    X_nb_val   = apply_weights(X_nb_val,   NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    X_nb_test  = apply_weights(X_nb_test,  NB_FEAT_PARAMS, nb_fit_info["block_cols"])
    g_cols = gaussian_col_indices(NB_FEAT_PARAMS)

    classes = state["classes"]
    K   = len(classes)
    n_tv = len(train_df) + len(val_df)
    print("Classes:", classes)
    print(f"Train / val / test rows  : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"LR/RF feature dim        : {X_train.shape[1]}")
    print(f"NB feature dim           : {X_nb_train.shape[1]}")

    # ── Stage 1: 5-fold CV for base model hyperparameter selection ────────────
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

    # ── Stage 2: OOF prediction generation on 80% trainval ───────────────────
    # Both df_tv and X_nb_tv cover the same rows in the same order, so a single
    # boolean mask applies to LR/RF and NB alike — no alignment bookkeeping.
    df_tv   = pd.concat([train_df, val_df], ignore_index=True)
    X_tv    = np.vstack([X_train, X_val])
    y_tv    = np.concatenate([y_train, y_val])
    X_nb_tv = np.vstack([X_nb_train, X_nb_val])
    y_nb_tv = np.concatenate([
        np.array([class_to_idx[c] for c in train_df["Painting"]]),
        np.array([class_to_idx[c] for c in val_df["Painting"]]),
    ])

    print(f"\n[Stage 2] OOF prediction generation on {n_tv}-row trainval pool...")

    rng      = np.random.RandomState(RANDOM_STATE)
    tv_ids   = df_tv["unique_id"].unique()
    rng.shuffle(tv_ids)
    id_folds = np.array_split(tv_ids, K_FOLDS)

    oof_lr = np.zeros((n_tv, K))
    oof_nb = np.zeros((n_tv, K))
    oof_rf = np.zeros((n_tv, K))

    for fold_i, fold_val_ids in enumerate(id_folds):
        fold_val_set   = set(fold_val_ids)
        fold_train_set = set(np.concatenate([id_folds[j] for j in range(K_FOLDS)
                                             if j != fold_i]))

        tr_mask = df_tv["unique_id"].isin(fold_train_set).values
        vl_mask = df_tv["unique_id"].isin(fold_val_set).values

        print(f"  Fold {fold_i+1}/{K_FOLDS}: train={tr_mask.sum()}  val={vl_mask.sum()}")

        # LR
        lr_f = LogisticRegression(
            max_iter=2500, C=best_lr_C, class_weight=LR_CLASS_WEIGHT,
            solver="lbfgs", random_state=RANDOM_STATE)
        lr_f.fit(X_tv[tr_mask], y_tv[tr_mask])
        oof_lr[vl_mask] = lr_f.predict_proba(X_tv[vl_mask])

        # NB — same mask as LR/RF (rows are in identical order)
        nb_f  = NaiveBayes(alpha=best_nb_alpha,  gaussian_cols=g_cols)
        cnb_f = CNaiveBayes(alpha=best_nb_alpha, gaussian_cols=g_cols)
        nb_f.fit(X_nb_tv[tr_mask],  y_nb_tv[tr_mask])
        cnb_f.fit(X_nb_tv[tr_mask], y_nb_tv[tr_mask])
        oof_nb[vl_mask] = _nb_proba(X_nb_tv[vl_mask], nb_f, cnb_f, best_nb_blend)

        # RF
        rf_f = RandomForestClassifier(
            n_estimators=best_rf_nest, max_depth=best_rf_depth,
            random_state=RANDOM_STATE, n_jobs=-1)
        rf_f.fit(X_tv[tr_mask], y_tv[tr_mask])
        oof_rf[vl_mask] = rf_f.predict_proba(X_tv[vl_mask])

    print(f"\n--- OOF base model performance ({n_tv} trainval rows) ---")
    _report("LR  (OOF)", y_tv, np.argmax(oof_lr, axis=1))
    _report("NB  (OOF)", y_tv, np.argmax(oof_nb, axis=1))
    _report("RF  (OOF)", y_tv, np.argmax(oof_rf, axis=1))

    meta_train = np.hstack([oof_lr, oof_nb, oof_rf])
    print(f"\n  Meta-feature matrix: {meta_train.shape}"
          f"  [p_lr(0..2) | p_nb(0..2) | p_rf(0..2)]")

    # ── Stage 3: Train meta-LR on OOF features ───────────────────────────────
    print(f"\n[Stage 3] Training meta-LR (C={META_C:.0e}) on 9-dim OOF features...")
    meta_lr = LogisticRegression(
        max_iter=5000, C=META_C, solver="lbfgs", random_state=RANDOM_STATE)
    meta_lr.fit(meta_train, y_tv)

    coef = meta_lr.coef_
    feat_names = ([f"LR_c{k}" for k in range(K)] +
                  [f"NB_c{k}" for k in range(K)] +
                  [f"RF_c{k}" for k in range(K)])
    short_cls = ["Persistence", "StarryNight", "WaterLily"]
    print(f"\n  Meta-LR coefficients (rows = predicted class, cols = base-model probs):")
    print(f"  {'':18s}" + "".join(f"{n:>10s}" for n in feat_names))
    for i, cname in enumerate(short_cls):
        print(f"  {cname:18s}" + "".join(f"{coef[i, j]:>10.3f}" for j in range(9)))

    # ── Stage 4 / Protocol B ─────────────────────────────────────────────────
    print(f"\n[Stage 4 / Protocol B] Refitting base models on 80%, evaluating on test...")

    lr80 = LogisticRegression(
        max_iter=2500, C=best_lr_C, class_weight=LR_CLASS_WEIGHT,
        solver="lbfgs", random_state=RANDOM_STATE)
    lr80.fit(X_tv, y_tv)
    p_lr_test = lr80.predict_proba(X_test)

    nb80  = NaiveBayes(alpha=best_nb_alpha,  gaussian_cols=g_cols)
    cnb80 = CNaiveBayes(alpha=best_nb_alpha, gaussian_cols=g_cols)
    nb80.fit(X_nb_tv,  y_nb_tv)
    cnb80.fit(X_nb_tv, y_nb_tv)
    p_nb_test = _nb_proba(X_nb_test, nb80, cnb80, best_nb_blend)

    rf80 = RandomForestClassifier(
        n_estimators=best_rf_nest, max_depth=best_rf_depth,
        random_state=RANDOM_STATE, n_jobs=-1)
    rf80.fit(X_tv, y_tv)
    p_rf_test = rf80.predict_proba(X_test)

    meta_test  = np.hstack([p_lr_test, p_nb_test, p_rf_test])
    pred_stack = meta_lr.predict(meta_test)

    pred_lr  = np.argmax(p_lr_test, axis=1)
    pred_nb  = np.argmax(p_nb_test, axis=1)
    pred_rf  = np.argmax(p_rf_test, axis=1)
    pred_maj = _majority_vote_3(pred_lr, pred_nb, pred_rf)

    print(f"\n--- Protocol B test results (n={len(y_test)}) ---")
    _report("LR  (base model)",          y_test, pred_lr)
    _report("NB  (base model)",          y_test, pred_nb)
    _report("RF  (base model)",          y_test, pred_rf)
    _report("Majority vote (3 models)",  y_test, pred_maj)
    print()
    _report("Stacking  (meta-LR, 9-dim)", y_test, pred_stack)

    print(f"\n  Confusion matrix — Stacking (meta-LR):")
    _print_cm(y_test, pred_stack)

    agree = (pred_stack == pred_maj).sum()
    print(f"\n  Stacking agrees with majority vote on {agree}/{len(y_test)}"
          f" ({100*agree/len(y_test):.1f}%) of test rows.")
    maj_wrong, stack_right = (pred_maj != y_test), (pred_stack == y_test)
    maj_right,  stack_wrong = (pred_maj == y_test), (pred_stack != y_test)
    print(f"  Overrides majority -> correct : {(maj_wrong & stack_right).sum()} cases")
    print(f"  Overrides majority -> wrong   : {(maj_right & stack_wrong).sum()} cases")


if __name__ == "__main__":
    main()
