"""
Single entry point for multiseed 80/20 stacking experiments.

Edit PRESETS below (or pass --preset NAME) to compare meta-LR C and RF settings
with fixed LR/NB. Same protocol as the old min-priority / seed-eval scripts.

Usage
-----
  python stacking_experiments.py              # run all PRESETS, print summary
  python stacking_experiments.py --preset A   # run one preset by id
  python stacking_experiments.py --rank       # after all, sort by min acc then mean

Preset fields
-------------
  id          short id for --preset
  meta_c      meta logistic regression C (smaller => stronger L2 on meta layer)
  rf_n_est, rf_depth, rf_min_leaf   RandomForest (depth None = unlimited)
  seeds       "short" (6) or "long" (10) — which split seeds to average over
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

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
LR_C = 100.0
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

SEEDS_SHORT = [1, 7, 13, 21, 42, 84]
SEEDS_LONG = [1, 7, 13, 21, 42, 84, 97, 123, 256, 512]


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    meta_c: float
    rf_n_est: int
    rf_depth: Optional[int]
    rf_min_leaf: int
    seeds: Literal["short", "long"]


# --- Edit / duplicate rows here to try new approaches ---------------------------------
PRESETS: Tuple[Preset, ...] = (
    Preset(
        "A",
        "Baseline: strong meta (C=1e4) + deep RF (200 trees, unbounded depth)",
        1e4,
        200,
        None,
        1,
        "long",
    ),
    Preset(
        "B",
        "Regularised RF: meta C=1, RF 400/depth12/leaf4",
        1.0,
        400,
        12,
        4,
        "long",
    ),
    Preset(
        "C",
        "Floor-focused: meta C=0.5, RF 200/unbounded/leaf1 (6-seed screen)",
        0.5,
        200,
        None,
        1,
        "short",
    ),
    Preset(
        "D",
        "Same RF as C but meta C=1.0 (6-seed screen)",
        1.0,
        200,
        None,
        1,
        "short",
    ),
    Preset(
        "E",
        "Match old fast-eval: meta 1, RF 400/12/4 (6-seed screen)",
        1.0,
        400,
        12,
        4,
        "short",
    ),
)


def _seed_list(which: Literal["short", "long"]) -> List[int]:
    return list(SEEDS_SHORT if which == "short" else SEEDS_LONG)


def _nb_proba(X, nb_m, cnb_m, blend: float):
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


def _oof_meta(
    df_fit, state, class_to_idx, nb_fit_info, g_cols, seed, rf_n_est, rf_depth, rf_min_leaf
):
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

        lr, nb, cnb, rf_m = _fit_base(
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
        oof_rf[vl_mask] = rf_m.predict_proba(X_fit[vl_mask])

    return np.hstack([oof_lr, oof_nb, oof_rf]), X_fit, y_fit, X_nb_fit, y_nb_fit


def eval_one_seed(
    df_clean: pd.DataFrame,
    seed: int,
    meta_c: float,
    rf_n_est: int,
    rf_depth,
    rf_min_leaf: int,
) -> Tuple[float, float]:
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

    lr, nb, cnb, rf_m = _fit_base(
        X_fit, y_fit, X_nb_fit, y_nb_fit, g_cols, seed, rf_n_est, rf_depth, rf_min_leaf
    )

    X_test, y_test = transform_df(test_df, state)
    X_nb_test, _ = build_features(test_df, fit_info=nb_fit_info, params=NB_FEAT_PARAMS)
    X_nb_test = apply_weights(X_nb_test, NB_FEAT_PARAMS, nb_fit_info["block_cols"])

    p_lr = lr.predict_proba(X_test)
    p_nb = _nb_proba(X_nb_test, nb, cnb, NB_BLEND)
    p_rf = rf_m.predict_proba(X_test)
    meta_test = np.hstack([p_lr, p_nb, p_rf])
    pred = meta_lr.predict(meta_test)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return acc, f1


def _summarize(accs: Sequence[float]) -> dict:
    a = np.array(accs, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p25": float(np.percentile(a, 25)),
    }


def run_preset(df_clean: pd.DataFrame, p: Preset) -> Tuple[dict, dict, List[float]]:
    seeds = _seed_list(p.seeds)
    accs, f1s = [], []
    for sd in seeds:
        acc, f1 = eval_one_seed(
            df_clean, sd, p.meta_c, p.rf_n_est, p.rf_depth, p.rf_min_leaf
        )
        accs.append(acc)
        f1s.append(f1)
    return _summarize(accs), _summarize(f1s), accs


def main():
    ap = argparse.ArgumentParser(description="Multiseed 80/20 stacking experiments.")
    ap.add_argument(
        "--preset",
        type=str,
        metavar="ID",
        help="Run only preset id (e.g. A, B, C). Default: run all.",
    )
    ap.add_argument(
        "--rank",
        action="store_true",
        help="After running, print presets ordered by higher min, then mean.",
    )
    args = ap.parse_args()

    df_clean = clean(pd.read_csv(CSV_PATH))
    presets = [pr for pr in PRESETS if pr.id == args.preset] if args.preset else list(PRESETS)
    if args.preset and not presets:
        ids = ", ".join(p.id for p in PRESETS)
        raise SystemExit(f"Unknown preset {args.preset!r}. Choose one of: {ids}")

    rows = []
    for p in presets:
        sa, sf, _ = run_preset(df_clean, p)
        rows.append((p, sa, sf))
        depth = "None" if p.rf_depth is None else str(p.rf_depth)
        print(
            f"[{p.id}] {p.label}\n"
            f"      meta_C={p.meta_c:g}  RF: n_est={p.rf_n_est} depth={depth} "
            f"min_leaf={p.rf_min_leaf}  seeds={p.seeds} (n={len(_seed_list(p.seeds))})\n"
            f"      acc  mean={sa['mean']:.4f} std={sa['std']:.4f} "
            f"min={sa['min']:.4f} p25={sa['p25']:.4f} max={sa['max']:.4f}\n"
            f"      macro_f1 mean={sf['mean']:.4f} min={sf['min']:.4f}\n"
        )

    if args.rank and len(rows) > 1:
        print("=== Ranked by (min acc, p25, mean, -std) ===")
        ranked = sorted(rows, key=lambda x: (-x[1]["min"], -x[1]["p25"], -x[1]["mean"], x[1]["std"]))
        for p, sa, _ in ranked:
            short = p.label if len(p.label) <= 52 else p.label[:49] + "..."
            print(f"  {p.id}  min={sa['min']:.4f}  mean={sa['mean']:.4f}  {short}")


if __name__ == "__main__":
    main()
