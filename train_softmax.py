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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from pipeline import (
    CSV_PATH,
    RANDOM_STATE,
    COL_TARGET,
    clean,
    grouped_kfold_split,
    fit_state,
    transform_df,
)

K_FOLDS  = 5
MAX_ITER = 2500

PAIR_A = "The Starry Night"
PAIR_B = "The Water Lily Pond"

# Main model search
MAX_FEATURES_CANDIDATES = [6000, 8000]
MIN_DF_CANDIDATES       = [1, 2]
C_CANDIDATES            = [10.0, 20.0, 50.0, 100.0]
CLASS_WEIGHT_OPTIONS    = [None, "balanced"]

# Resolver search
PAIR_C_CANDIDATES      = [1.0, 5.0, 10.0]
PAIR_MARGIN_CANDIDATES = [0.08, 0.12, 0.16, 0.20]


def softmax(logits):
    z  = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def fit_pair_resolver(X_train, y_train, X_val, y_val, classes):
    if PAIR_A not in classes or PAIR_B not in classes:
        return None
    idx_a = classes.index(PAIR_A)
    idx_b = classes.index(PAIR_B)
    tr_mask = (y_train == idx_a) | (y_train == idx_b)
    va_mask = (y_val   == idx_a) | (y_val   == idx_b)
    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        return None

    Xp_tr = X_train[tr_mask]
    yp_tr = (y_train[tr_mask] == idx_b).astype(int)
    Xp_va = X_val[va_mask]
    y_va_full = y_val[va_mask]

    best     = None
    best_acc = -1.0
    for c_val in PAIR_C_CANDIDATES:
        m = LogisticRegression(max_iter=MAX_ITER, C=c_val,
                               solver="lbfgs", random_state=RANDOM_STATE)
        m.fit(Xp_tr, yp_tr)
        pred      = (m.predict_proba(Xp_va)[:, 1] >= 0.5).astype(int)
        pred_full = np.where(pred == 1, idx_b, idx_a)
        acc = float(accuracy_score(y_va_full, pred_full))
        if acc > best_acc:
            best     = {"model": m, "pair_C": c_val,
                        "idx_a": idx_a, "idx_b": idx_b, "acc": acc}
            best_acc = acc
    return best


def apply_pair_resolver(main_probs, main_pred, pair_info, X, margin_thr):
    if pair_info is None:
        return main_pred.copy(), 0
    idx_a = pair_info["idx_a"]
    idx_b = pair_info["idx_b"]
    top2    = np.argsort(main_probs, axis=1)[:, -2:]
    margins = np.sort(main_probs, axis=1)[:, -1] - np.sort(main_probs, axis=1)[:, -2]
    cand = np.array(
        [({idx_a, idx_b} == set(top2[i])) and (margins[i] <= margin_thr)
         for i in range(len(main_pred))],
        dtype=bool,
    )
    out = main_pred.copy()
    n   = int(cand.sum())
    if n == 0:
        return out, 0
    p_b     = pair_info["model"].predict_proba(X[cand])[:, 1]
    out[cand] = np.where(p_b >= 0.5, idx_b, idx_a)
    return out, n


def evaluate_with_optional_resolver(model, pair_info, X_val, y_val,
                                    X_ref_for_resolver):
    logits = X_val @ model.coef_.T + model.intercept_
    probs  = softmax(logits)
    pred   = np.argmax(probs, axis=1)
    base_acc = float(accuracy_score(y_val, pred))
    base_f1  = float(f1_score(y_val, pred, average="macro"))

    best = {"acc": base_acc, "f1": base_f1, "thr": None, "overrides": 0, "pred": pred}
    if pair_info is None:
        return best

    for thr in PAIR_MARGIN_CANDIDATES:
        p2, n = apply_pair_resolver(probs, pred, pair_info, X_ref_for_resolver, thr)
        acc = float(accuracy_score(y_val, p2))
        f1  = float(f1_score(y_val, p2, average="macro"))
        if acc > best["acc"]:
            best = {"acc": acc, "f1": f1, "thr": thr, "overrides": n, "pred": p2}
    return best


def save_final_state_and_model(state, model, pair_meta):
    np.savez(
        "preprocess_state.npz",
        num_means      = state["num_means"],
        num_stds       = state["num_stds"],
        num_clips      = state["num_clips"],
        idf            = state["idf"],
        price_mean     = np.array([state["price_mean"]]),
        price_std      = np.array([state["price_std"]]),
        price_clip     = np.array([state["price_clip"]]),
        num_medians    = state["num_medians"],
        likert_medians = state["likert_medians"],
        price_median   = np.array([state["price_median"]]),
    )
    with open("preprocess_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes":         state["classes"],
                "vocab":           state["vocab"],
                "room_cats":       state["room_cats"],
                "who_cats":        state["who_cats"],
                "season_cats":     state["season_cats"],
                "clip_percentile": 97,
                "max_features":    state["max_features"],
                "min_df":          state["min_df"],
                "impute":          "none",
                "text_normalization":
                    "lowercase + remove non-alnum + collapse spaces",
                "pair_resolver":   pair_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.savez(
        "model_params.npz",
        W          = model.coef_.T.astype(float),
        b          = model.intercept_.astype(float),
        pair_w     = np.array(pair_meta["pair_w"], dtype=float)
                     if pair_meta["enabled"] else np.array([]),
        pair_b     = np.array([pair_meta["pair_b"]], dtype=float)
                     if pair_meta["enabled"] else np.array([]),
        pair_idx   = np.array(pair_meta["pair_idx"], dtype=int)
                     if pair_meta["enabled"] else np.array([], dtype=int),
        pair_margin = np.array([pair_meta["margin"]], dtype=float)
                     if pair_meta["enabled"] else np.array([-1.0], dtype=float),
    )


def main():
    df       = pd.read_csv(CSV_PATH)
    df_clean = clean(df)

    grid = [
        {"max_features": mf, "min_df": md, "C": c_val, "class_weight": cw}
        for mf in MAX_FEATURES_CANDIDATES
        for md in MIN_DF_CANDIDATES
        for c_val in C_CANDIDATES
        for cw in CLASS_WEIGHT_OPTIONS
    ]

    all_results  = []
    best         = None
    best_val_acc = -1.0

    for params in grid:
        fold_metrics = []
        for fold_idx, (tr, va, te) in enumerate(
            grouped_kfold_split(df_clean, k=K_FOLDS, random_state=RANDOM_STATE),
            start=1,
        ):
            state        = fit_state(tr, max_features=params["max_features"],
                                     min_df=params["min_df"])
            Xtr, ytr     = transform_df(tr, state)
            Xva, yva     = transform_df(va, state)
            Xte, yte     = transform_df(te, state)

            main_model = LogisticRegression(
                max_iter=MAX_ITER, C=params["C"],
                class_weight=params["class_weight"],
                solver="lbfgs", random_state=RANDOM_STATE,
            )
            main_model.fit(Xtr, ytr)

            pair_info = fit_pair_resolver(Xtr, ytr, Xva, yva, state["classes"])
            val_eval  = evaluate_with_optional_resolver(
                main_model, pair_info, Xva, yva, Xva)
            test_eval = evaluate_with_optional_resolver(
                main_model, pair_info, Xte, yte, Xte)

            fold_metrics.append({
                "fold":               fold_idx,
                "val_acc":            val_eval["acc"],
                "val_f1":             val_eval["f1"],
                "test_acc":           test_eval["acc"],
                "test_f1":            test_eval["f1"],
                "pair_thr":           val_eval["thr"],
                "pair_overrides_val": val_eval["overrides"],
            })

        result = {
            "params":        params,
            "val_acc_mean":  float(np.mean([f["val_acc"]  for f in fold_metrics])),
            "val_acc_std":   float(np.std( [f["val_acc"]  for f in fold_metrics])),
            "val_f1_mean":   float(np.mean([f["val_f1"]   for f in fold_metrics])),
            "test_acc_mean": float(np.mean([f["test_acc"] for f in fold_metrics])),
            "test_f1_mean":  float(np.mean([f["test_f1"]  for f in fold_metrics])),
            "folds":         fold_metrics,
        }
        all_results.append(result)
        if result["val_acc_mean"] > best_val_acc:
            best         = result
            best_val_acc = result["val_acc_mean"]

    if best is None:
        raise RuntimeError("No best config selected.")

    # Final model on full cleaned data
    final_state = fit_state(df_clean,
                            max_features=best["params"]["max_features"],
                            min_df=best["params"]["min_df"])
    X_full, y_full = transform_df(df_clean, final_state)
    final_model = LogisticRegression(
        max_iter=MAX_ITER, C=best["params"]["C"],
        class_weight=best["params"]["class_weight"],
        solver="lbfgs", random_state=RANDOM_STATE,
    )
    final_model.fit(X_full, y_full)

    # Pair resolver on full data for deployment
    pair_meta = {"enabled": False}
    if PAIR_A in final_state["classes"] and PAIR_B in final_state["classes"]:
        idx_a = final_state["classes"].index(PAIR_A)
        idx_b = final_state["classes"].index(PAIR_B)
        mask  = (y_full == idx_a) | (y_full == idx_b)
        if mask.sum() > 0:
            Xp = X_full[mask]
            yp = (y_full[mask] == idx_b).astype(int)
            m_pair = LogisticRegression(
                max_iter=MAX_ITER, C=5.0,
                solver="lbfgs", random_state=RANDOM_STATE,
            )
            m_pair.fit(Xp, yp)
            selected_rows = [r for r in all_results if r["params"] == best["params"]]
            thrs   = [f["pair_thr"] for f in selected_rows[0]["folds"]
                      if f["pair_thr"] is not None]
            margin = float(np.median(thrs)) if thrs else -1.0
            pair_meta = {
                "enabled":  margin > 0,
                "pair_idx": [idx_a, idx_b],
                "pair_w":   m_pair.coef_.ravel().tolist(),
                "pair_b":   float(m_pair.intercept_[0]),
                "margin":   margin,
                "pair_C":   5.0,
            }

    save_final_state_and_model(final_state, final_model, pair_meta)

    summary = {
        "best_by_grouped_cv": best,
        "top5": sorted(all_results, key=lambda x: x["val_acc_mean"], reverse=True)[:5],
        "final_training": {
            "n_rows":                int(len(df_clean)),
            "n_features":            int(X_full.shape[1]),
            "classes":               final_state["classes"],
            "pair_resolver_enabled": pair_meta["enabled"],
        },
    }
    with open("cv_selection_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Best grouped-CV params:", best["params"])
    print(
        f"Grouped-CV val_acc={best['val_acc_mean']:.4f}"
        f"\u00b1{best['val_acc_std']:.4f}, "
        f"val_f1={best['val_f1_mean']:.4f}"
    )
    print(
        f"Grouped-CV test_acc_mean={best['test_acc_mean']:.4f}, "
        f"test_f1_mean={best['test_f1_mean']:.4f}"
    )
    print("Saved preprocess_state.*, model_params.npz, cv_selection_metrics.json")


if __name__ == "__main__":
    main()
