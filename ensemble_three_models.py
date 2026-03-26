"""
Ensemble: Softmax (multinomial LR) + Multinomial NB (text TF-IDF) + Random Forest.

Uses one shared pipeline so results are comparable:
  - training_data.csv + preprocessing.clean()
  - Person-level 60/20/20 split (data_splitting.regular_split, same seed as below)
  - Feature matrix from train_softmax.fit_state / transform_df (matches train_softmax.py)

NB is fit only on the non-negative L2-normalized TF-IDF block (same vocab as the LR/RF
features). LR and RF use the full stacked feature vector.

Validation: grid-search nonnegative weights w_lr + w_nb + w_rf = 1 on val probabilities.

Rule-based fusion (val-tuned, test-reported):
  - Majority vote among LR / NB / RF.
  - LR when max(softmax prob) ≥ τ else majority vote (τ tuned on val).
  - Same with margin (top1 − top2) ≥ m as an OR condition so confident borderline cases stay LR.
  - LR when confident else RF only (RF is strongest single model besides refit LR).
  - LR when confident else argmax of average of NB+RF class probabilities (text + structured trees).

Test: we report two regimes so you do not confuse this script with train_softmax.py:
  (1) Models trained on 60% train only — comparable to "student A uses train, B uses val"
      for stacking weights; LR test here is often ~0.89 even though grouped-CV prints ~0.92,
      because grouped CV trains each fold on ~64% of people (4/5 of 80%), not 60%.
  (2) After picking weights, refit all three models on train+val (80%) with the same
      fitted preprocess state (vocab from train), then test — closer to a final model and
      fairer vs your high softmax CV / full-data numbers.

Requires: scikit-learn, numpy, pandas.
Run: python ensemble_three_models.py
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from data_splitting import regular_split
from train_softmax import CSV_PATH, RANDOM_STATE, clean, fit_state, transform_df

# Match the grouped-CV-selected softmax setup (see train_softmax.py search grid outcome)
MAX_FEATURES = 6000
MIN_DF = 1
LR_C = 100.0
LR_CLASS_WEIGHT = None

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_RANDOM_STATE = 42

NB_ALPHA = 1.0

WEIGHT_STEPS = 11  # 0.0, 0.1, ..., 1.0

# Rule fusion: τ grid (LR confidence gate); margin grid for LR top1−top2
TAU_GRID = np.linspace(0.34, 0.995, 48)
MARGIN_GRID = np.linspace(0.01, 0.45, 23)


def _majority_vote_3(pred_lr: np.ndarray, pred_nb: np.ndarray, pred_rf: np.ndarray) -> np.ndarray:
    """Hard majority over three integer predictions; ties → LR, then RF, then NB."""
    stack = np.stack([pred_lr, pred_nb, pred_rf], axis=1)
    n = stack.shape[0]
    out = np.empty(n, dtype=int)
    pref = (0, 2, 1)  # LR, RF, NB
    for i in range(n):
        row = stack[i]
        vals, counts = np.unique(row, return_counts=True)
        if counts.max() >= 2:
            out[i] = int(vals[np.argmax(counts)])
            continue
        for j in pref:
            out[i] = int(row[j])
            break
    return out


def _lr_confidence_and_margin(p_lr: np.ndarray):
    conf = p_lr.max(axis=1)
    s = np.sort(p_lr, axis=1)
    margin = s[:, -1] - s[:, -2]
    return conf, margin


def _tune_rule_on_val(
    p_lr_val,
    p_nb_val,
    p_rf_val,
    y_val,
    rule: str,
):
    """
    rule in:
      'maj' — majority only (no tuning)
      'lr_tau_else_maj'
      'lr_tau_or_margin_else_maj'
      'lr_tau_else_rf'
      'lr_tau_else_nb_rf_soft'
    Returns (best_kwargs dict, val_acc, val_f1) for rules that need tuning; maj returns ({}, acc, f1).
    """
    plr = np.argmax(p_lr_val, axis=1)
    pnb = np.argmax(p_nb_val, axis=1)
    prf = np.argmax(p_rf_val, axis=1)
    maj = _majority_vote_3(plr, pnb, prf)

    if rule == "maj":
        acc = accuracy_score(y_val, maj)
        f1v = f1_score(y_val, maj, average="macro", zero_division=0)
        return {}, acc, f1v

    conf, margin = _lr_confidence_and_margin(p_lr_val)
    best = None
    best_acc = -1.0

    if rule == "lr_tau_else_maj":
        for tau in TAU_GRID:
            pred = np.where(conf >= tau, plr, maj)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best = {"tau": float(tau)}
    elif rule == "lr_tau_or_margin_else_maj":
        for tau in TAU_GRID:
            for m in MARGIN_GRID:
                use_lr = (conf >= tau) | (margin >= m)
                pred = np.where(use_lr, plr, maj)
                acc = accuracy_score(y_val, pred)
                if acc > best_acc:
                    best_acc = acc
                    best = {"tau": float(tau), "margin": float(m)}
    elif rule == "lr_tau_else_rf":
        for tau in TAU_GRID:
            pred = np.where(conf >= tau, plr, prf)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best = {"tau": float(tau)}
    elif rule == "lr_tau_else_nb_rf_soft":
        p_soft = (p_nb_val + p_rf_val) / 2.0
        ps = np.argmax(p_soft, axis=1)
        for tau in TAU_GRID:
            pred = np.where(conf >= tau, plr, ps)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best = {"tau": float(tau)}
    else:
        raise ValueError(rule)

    pred_best = None
    if rule == "lr_tau_else_maj":
        pred_best = np.where(conf >= best["tau"], plr, maj)
    elif rule == "lr_tau_or_margin_else_maj":
        use_lr = (conf >= best["tau"]) | (margin >= best["margin"])
        pred_best = np.where(use_lr, plr, maj)
    elif rule == "lr_tau_else_rf":
        pred_best = np.where(conf >= best["tau"], plr, prf)
    elif rule == "lr_tau_else_nb_rf_soft":
        p_soft = (p_nb_val + p_rf_val) / 2.0
        ps = np.argmax(p_soft, axis=1)
        pred_best = np.where(conf >= best["tau"], plr, ps)

    f1v = f1_score(y_val, pred_best, average="macro", zero_division=0)
    return best, best_acc, f1v


def _apply_rule_on_test(
    p_lr_te,
    p_nb_te,
    p_rf_te,
    rule: str,
    kwargs: dict,
):
    plr = np.argmax(p_lr_te, axis=1)
    pnb = np.argmax(p_nb_te, axis=1)
    prf = np.argmax(p_rf_te, axis=1)
    maj = _majority_vote_3(plr, pnb, prf)
    if rule == "maj":
        return maj
    conf, margin = _lr_confidence_and_margin(p_lr_te)
    if rule == "lr_tau_else_maj":
        return np.where(conf >= kwargs["tau"], plr, maj)
    if rule == "lr_tau_or_margin_else_maj":
        use_lr = (conf >= kwargs["tau"]) | (margin >= kwargs["margin"])
        return np.where(use_lr, plr, maj)
    if rule == "lr_tau_else_rf":
        return np.where(conf >= kwargs["tau"], plr, prf)
    if rule == "lr_tau_else_nb_rf_soft":
        p_soft = (p_nb_te + p_rf_te) / 2.0
        ps = np.argmax(p_soft, axis=1)
        return np.where(conf >= kwargs["tau"], plr, ps)
    raise ValueError(rule)


def _tfidf_slice(X, n_vocab: int):
    if n_vocab <= 0 or n_vocab > X.shape[1]:
        raise ValueError("Invalid TF-IDF width for feature matrix.")
    return X[:, -n_vocab:]


def _proba_grid_search(p_lr, p_nb, p_rf, y_true):
    """Return (best_weights_tuple, best_acc, best_f1) on validation rows."""
    steps = np.linspace(0.0, 1.0, WEIGHT_STEPS)
    best = None
    best_acc = -1.0
    for w_lr, w_nb in product(steps, steps):
        w_rf = 1.0 - w_lr - w_nb
        if w_rf < -1e-9:
            continue
        p = w_lr * p_lr + w_nb * p_nb + w_rf * p_rf
        pred = np.argmax(p, axis=1)
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if acc > best_acc:
            best_acc = acc
            best = ((float(w_lr), float(w_nb), float(max(w_rf, 0.0))), acc, f1)
    return best


def main():
    df = pd.read_csv(CSV_PATH)
    df_clean = clean(df)
    train_df, val_df, test_df = regular_split(df_clean, random_state=RANDOM_STATE)

    state = fit_state(train_df, max_features=MAX_FEATURES, min_df=MIN_DF)
    X_train, y_train = transform_df(train_df, state)
    X_val, y_val = transform_df(val_df, state)
    X_test, y_test = transform_df(test_df, state)

    n_vocab = len(state["vocab"])
    X_train_text = _tfidf_slice(X_train, n_vocab)
    X_val_text = _tfidf_slice(X_val, n_vocab)
    X_test_text = _tfidf_slice(X_test, n_vocab)

    # --- Model 1: multinomial logistic regression (softmax) ---
    lr = LogisticRegression(
        max_iter=2500,
        C=LR_C,
        class_weight=LR_CLASS_WEIGHT,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    p_lr_val = lr.predict_proba(X_val)
    p_lr_test = lr.predict_proba(X_test)

    # --- Model 2: Multinomial NB on text TF-IDF only ---
    nb = MultinomialNB(alpha=NB_ALPHA)
    nb.fit(X_train_text, y_train)
    p_nb_val = nb.predict_proba(X_val_text)
    p_nb_test = nb.predict_proba(X_test_text)

    # --- Model 3: Random forest on full features ---
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    p_rf_val = rf.predict_proba(X_val)
    p_rf_test = rf.predict_proba(X_test)

    classes = state["classes"]
    print("Classes (order = label index 0..K-1):", classes)
    print(f"Train / val / test rows: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Feature dim = {X_train.shape[1]} (TF-IDF cols = {n_vocab})\n")

    def report(name, y_true, pred):
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        print(f"  {name:18s}  acc={acc:.4f}  macro_f1={f1:.4f}")

    print("--- Single models (val) ---")
    report("LR (softmax)", y_val, np.argmax(p_lr_val, axis=1))
    report("NB (text)", y_val, np.argmax(p_nb_val, axis=1))
    report("RandomForest", y_val, np.argmax(p_rf_val, axis=1))

    print("\n--- Single models (test) ---")
    pred_lr_test = np.argmax(p_lr_test, axis=1)
    pred_nb_test = np.argmax(p_nb_test, axis=1)
    pred_rf_test = np.argmax(p_rf_test, axis=1)
    report("LR (softmax)", y_test, pred_lr_test)
    report("NB (text)", y_test, pred_nb_test)
    report("RandomForest", y_test, pred_rf_test)

    print("\n--- Protocol A (60/20/20): test metrics ---")
    # Keep only:
    #  1) Majority vote (LR / NB / RF)
    #  2) LR high-confidence gate → LR else Majority (tau tuned on val)
    rules = [
        ("maj", "Majority vote (LR/NB/RF)"),
        ("lr_tau_else_maj", "LR high-confidence gate → LR else Majority"),
    ]
    kw_lr_else_maj = None
    for rule_key, label in rules:
        kw, _, _ = _tune_rule_on_val(p_lr_val, p_nb_val, p_rf_val, y_val, rule_key)
        if rule_key == "lr_tau_else_maj":
            kw_lr_else_maj = kw
        pred_te = _apply_rule_on_test(p_lr_test, p_nb_test, p_rf_test, rule_key, kw)
        tacc = accuracy_score(y_test, pred_te)
        tf1 = f1_score(y_test, pred_te, average="macro", zero_division=0)
        print(f"  {label}")
        print(f"    test acc={tacc:.4f}  macro_f1={tf1:.4f}")

    # --- Refit on 80% (train+val), same state (vocab fit on train only) ---
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    X_tv_text = _tfidf_slice(X_tv, n_vocab)

    lr80 = LogisticRegression(
        max_iter=2500,
        C=LR_C,
        class_weight=LR_CLASS_WEIGHT,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr80.fit(X_tv, y_tv)
    p_lr_t = lr80.predict_proba(X_test)

    nb80 = MultinomialNB(alpha=NB_ALPHA)
    nb80.fit(X_tv_text, y_tv)
    p_nb_t = nb80.predict_proba(X_test_text)

    rf80 = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf80.fit(X_tv, y_tv)
    p_rf_t = rf80.predict_proba(X_test)

    print("\n--- Protocol B (80/20): test metrics ---")
    plr80 = np.argmax(p_lr_t, axis=1)
    pnb80 = np.argmax(p_nb_t, axis=1)
    prf80 = np.argmax(p_rf_t, axis=1)

    maj80 = _majority_vote_3(plr80, pnb80, prf80)
    report("Majority vote (LR/NB/RF)", y_test, maj80)
    if kw_lr_else_maj:
        conf80, _ = _lr_confidence_and_margin(p_lr_t)
        gated80 = np.where(conf80 >= kw_lr_else_maj["tau"], plr80, maj80)
        report("LR high-confidence gate → LR else Majority", y_test, gated80)


if __name__ == "__main__":
    main()
