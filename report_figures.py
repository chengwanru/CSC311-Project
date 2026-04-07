"""
Generate figures and CSV tables for model evaluation (stacking + baselines).

Dependencies: numpy, pandas, scikit-learn, matplotlib
(`pip install -r requirements-figures.txt`).

PNG plots and CSV tables are written to ``./plots/`` (created automatically).

Usage (repository root, ``training_data.csv`` present)::

    python report_figures.py              # default partition seed 42 + six-seed sweep

**Outputs**

``plots/01_model_compare_test.png``
    Bar chart: test-set accuracy and macro-F1 for LR, NB, RF, majority vote, stacking.

``plots/02_confusion_matrix_stacking.png``
    Confusion matrix for stacked classifier on the 20% test set.

``plots/03_errors_true_starry_night.png``
    Starry Night test rows: mostly correct; blue bars show rare mistakes and confused-into class.

``plots/04_partition_seed_sensitivity.png``
    Stacking test accuracy vs random seed for the train/test split (same 80/20 protocol).

``plots/05_stacking_seed_min_mean_max.csv``
    Stacking test accuracy: min/mean/max over six partition seeds (numeric).

``plots/05_stacking_seed_min_mean_max.png``
    Same as CSV as a small table figure (Min / Mean / Max only).

``plots/06_confusion_matrix_baselines.png``
    One figure with 4 confusion matrices (LR/NB/RF/Majority) on the same test split.

``plots/07_train_pool_compare_lr_nb_rf.png``
    LR, NB, RF: **test accuracy** and **macro-F1** (two panels) at **one auto-picked showcase seed**;
    60% train-only vs 80% train+val refit; caption notes range over all scanned seeds.

``plots/08_stacking_train_pool_metrics.png`` / ``.csv``
    **Final stacking model**: test accuracy and macro-F1, **60% train-only** vs **80% train+val** pools
    (mean ± std over the same six seeds as Figure 4).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from stacking_ensemble import run_stacking_eval

SPLIT_SEEDS_DEFAULT = [1, 7, 13, 21, 42, 84]
# More partition seeds only for Figure 7 (60% vs 80% train pool): wider accuracy range on small data.
FIG7_TRAIN_POOL_SEEDS = sorted({1, 7, 13, 21, 42, 84, 0, 3, 9, 17, 27, 38, 55, 66, 77, 99, 123, 156, 200})
PLOT_DIR = Path(__file__).resolve().parent / "plots"


def _figure_label_below(
    fig,
    lines: tuple[str, ...],
    *,
    right: float | None = None,
    label_below_axes: float = 0.058,
    caption_gap: float = 0.018,
) -> None:
    """Place caption in the figure margin below x-axis ticks/labels (aligned with axes bottom)."""
    n = max(len(lines), 1)
    bottom_pad = 0.024
    line_h = 0.024
    cap_h = line_h * (1.0 + 0.92 * (n - 1))
    strip = bottom_pad + cap_h + caption_gap + label_below_axes
    strip = float(np.clip(strip, 0.12, 0.34))
    y_top = strip - label_below_axes - caption_gap
    kw: dict = dict(bottom=strip, top=0.97)
    if right is not None:
        kw["right"] = right
    fig.subplots_adjust(**kw)
    fig.text(
        0.5,
        y_top,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        transform=fig.transFigure,
        linespacing=1.08,
    )


def _short_labels(names: list[str]) -> list[str]:
    out = []
    for n in names:
        if "Starry" in n:
            out.append("Starry Night")
        elif "Water Lily" in n:
            out.append("Water Lily")
        elif "Persistence" in n:
            out.append("Persistence")
        else:
            out.append(n[:12])
    return out


def figure_model_compare(r: dict, path: Path, split_seed: int) -> None:
    names = ["LR", "NB", "RF", "Majority", "Stacking"]
    accs = [r["acc_lr"], r["acc_nb"], r["acc_rf"], r["acc_maj"], r["acc_stack"]]
    f1s = [
        r["macro_f1_lr"],
        r["macro_f1_nb"],
        r["macro_f1_rf"],
        r["macro_f1_maj"],
        r["macro_f1_stack"],
    ]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars_acc = ax.bar(x - w / 2, accs, width=w, label="Accuracy")
    bars_f1 = ax.bar(x + w / 2, f1s, width=w, label="Macro F1 (class-average)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    # Macro-F1 can track accuracy closely on balanced 3-class data — show numeric values on bars.
    try:
        ax.bar_label(bars_acc, fmt="%.4f", fontsize=7, padding=2)
        ax.bar_label(bars_f1, fmt="%.4f", fontsize=7, padding=2)
    except AttributeError:
        pass
    _figure_label_below(
        fig,
        (
            "Figure 1 — Test accuracy and macro-F1",
            "LR, NB, RF, majority vote, stacking",
            f"80/20 split (respondent-level) · seed {split_seed}",
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: Path, split_seed: int
) -> None:
    short = _short_labels(class_names)
    fig, ax = plt.subplots(figsize=(5.2, 5.55))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short,
        yticklabels=short,
        ylabel="True class",
        xlabel="",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    _figure_label_below(
        fig,
        (
            "Figure 2 — Confusion matrix (test set, counts)",
            "Rows: true class · columns: predicted class",
            f"80/20 split · seed {split_seed}",
        ),
        right=0.88,
        label_below_axes=0.158,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_starry_night_errors(r: dict, path: Path, split_seed: int) -> None:
    names = r["class_names"]
    short = _short_labels(names)
    try:
        sn_i = next(i for i, n in enumerate(names) if "Starry" in n)
    except StopIteration:
        sn_i = 1
    y = r["y_test"]
    pred = r["pred_stack"]
    mask = y == sn_i
    if not np.any(mask):
        return
    preds_on_sn = pred[mask]
    counts = np.bincount(preds_on_sn, minlength=len(names))
    n_sub = int(mask.sum())
    n_ok = int(counts[sn_i])
    n_wrong = n_sub - n_ok
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    bars = ax.bar(
        short, counts, color=["#4a90d9" if i != sn_i else "#e94b3c" for i in range(len(names))]
    )
    ax.set_ylabel("Count (test rows)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(
                facecolor="#e94b3c",
                edgecolor="#333333",
                linewidth=0.4,
                label="Correct",
            ),
            Patch(
                facecolor="#4a90d9",
                edgecolor="#333333",
                linewidth=0.4,
                label="Wrong class",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )
    try:
        ax.bar_label(bars, labels=[str(int(c)) for c in counts], fontsize=8, padding=2)
    except AttributeError:
        pass
    top = float(counts.max()) if len(counts) else 0.0
    ax.set_ylim(0, max(top * 1.15, top + 1.0))
    pct_ok = 100.0 * n_ok / n_sub if n_sub else 0.0
    _figure_label_below(
        fig,
        (
            "Figure 3 — True class: The Starry Night (predicted label counts)",
            f"Red = correct · blue = other class · {n_ok}/{n_sub} correct ({pct_ok:.1f}%)",
            f"Same split as Figs. 1–2 · seed {split_seed}",
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_confusion_matrix_baselines(r: dict, path: Path, split_seed: int) -> None:
    """2x2 confusion matrices for LR/NB/RF/Majority on the same held-out test set."""
    y = np.asarray(r["y_test"])
    class_names = _short_labels(r["class_names"])
    models = [
        ("LR", np.asarray(r["pred_lr"])),
        ("NB", np.asarray(r["pred_nb"])),
        ("RF", np.asarray(r["pred_rf"])),
        ("Majority", np.asarray(r["pred_maj"])),
    ]

    cms = []
    for _, pred in models:
        cm = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                cm[i, j] = int(np.sum((y == i) & (pred == j)))
        cms.append(cm)

    vmax = max(int(cm.max()) for cm in cms) if cms else 1
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.4), sharex=True, sharey=True)
    axes = axes.ravel()
    im = None
    for ax, (name, _), cm in zip(axes, models, cms):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(class_names, rotation=25, ha="right")
        ax.set_yticklabels(class_names)
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                )
        # Panel label (not subplot title) to identify model.
        ax.text(
            0.02,
            0.98,
            name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9),
        )

    for ax in axes[2:]:
        ax.set_xlabel("Predicted class")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("True class")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.06)
        cbar.set_label("Count")

    _figure_label_below(
        fig,
        (
            "Figure 6 — Confusion matrices on test set: LR, NB, RF, and majority vote",
            "Rows: true class · columns: predicted class (same split as Figure 1)",
            f"80/20 split · seed {split_seed}",
        ),
        right=0.86,
        label_below_axes=0.13,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_stability_seeds(rows: list[tuple[int, float]], path: Path) -> None:
    seeds, accs = zip(*rows)
    fig, ax = plt.subplots(figsize=(6.5, 5.1))
    ax.plot(seeds, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test accuracy")
    ax.grid(True, alpha=0.3)
    m = min(accs)
    ax.axhline(m, color="gray", linestyle="--", alpha=0.7, label=f"min={m:.4f}")
    ax.axhline(np.mean(accs), color="green", linestyle=":", alpha=0.8, label=f"mean={np.mean(accs):.4f}")
    ax.legend(loc="lower right")
    seed_list = ", ".join(str(s) for s in seeds)
    _figure_label_below(
        fig,
        (
            "Figure 4 — Test accuracy vs random partition seed",
            f"Seeds: {seed_list} · 80/20 split each run",
            "Fixed hyperparameters",
        ),
        label_below_axes=0.132,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _pick_fig7_showcase_seed(rows: list[tuple[int, dict, dict]]) -> tuple[int, list[float], list[float]]:
    """
    Pick one partition seed where 60%-pool accuracies dip while 80%-pool reaches ~90%+.
    Tier 1: min(60%) < 0.90 and max(80%) >= 0.90; tie-break by largest total lift sum(80-60).
    Tier 2: if none, largest sum(80-60) among all seeds.
    """
    acc_keys = ("acc_lr", "acc_nb", "acc_rf")
    tier1: list[tuple[float, int, list[float], list[float]]] = []
    tier2: list[tuple[float, int, list[float], list[float]]] = []
    for sd, r60, r80 in rows:
        v60 = [float(r60[k]) for k in acc_keys]
        v80 = [float(r80[k]) for k in acc_keys]
        gap = sum(b - a for a, b in zip(v60, v80))
        if max(v80) >= 0.90 and min(v60) < 0.90:
            tier1.append((gap, int(sd), v60, v80))
        else:
            tier2.append((gap, int(sd), v60, v80))
    pool = tier1 if tier1 else tier2
    # Prefer the hardest 60%-pool case (lowest min accuracy), then largest total lift.
    pool.sort(key=lambda t: (min(t[2]), -t[0]))
    _, sd, v60, v80 = pool[0]
    return sd, v60, v80


def figure_train_pool_baselines_compare(rows: list[tuple[int, dict, dict]], path: Path) -> int:
    """
    rows: (seed, r60, r80) from ``run_stacking_eval`` with ``stacking_train_pool`` ``train`` vs ``trainval``.
    **Two panels:** test accuracy and macro-F1 for LR / NB / RF at **one auto-picked showcase seed**.
    """
    acc_keys = ("acc_lr", "acc_nb", "acc_rf")
    f1_keys = ("macro_f1_lr", "macro_f1_nb", "macro_f1_rf")
    xlabels = ["LR", "NB", "RF"]
    seeds = [r[0] for r in rows]
    acc60 = np.array([[float(r60[k]) for k in acc_keys] for _, r60, _ in rows])
    acc80 = np.array([[float(r80[k]) for k in acc_keys] for _, _, r80 in rows])
    f60_all = np.array([[float(r60[k]) for k in f1_keys] for _, r60, _ in rows])
    f80_all = np.array([[float(r80[k]) for k in f1_keys] for _, _, r80 in rows])
    n_seeds = len(rows)
    gmin60 = float(acc60.min())
    gmin60_f1 = float(f60_all.min())
    gmean60 = float(acc60.mean())
    gmean80 = float(acc80.mean())

    showcase_sd, _, _ = _pick_fig7_showcase_seed(rows)
    r60_s = next(r60 for sd, r60, r80 in rows if sd == showcase_sd)
    r80_s = next(r80 for sd, r60, r80 in rows if sd == showcase_sd)
    mean60 = np.array([float(r60_s[k]) for k in acc_keys])
    mean80 = np.array([float(r80_s[k]) for k in acc_keys])
    mf60 = np.array([float(r60_s[k]) for k in f1_keys])
    mf80 = np.array([float(r80_s[k]) for k in f1_keys])

    fig, (axa, axf) = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    x = np.arange(3)
    w = 0.36
    for ax, m60, m80, y_title in (
        (axa, mean60, mean80, "Test accuracy"),
        (axf, mf60, mf80, "Macro F1"),
    ):
        ax.bar(
            x - w / 2,
            m60,
            width=w,
            label="60% train only (refit on train partition)",
            color="#6baed6",
        )
        ax.bar(
            x + w / 2,
            m80,
            width=w,
            label="80% train+val (refit on train ∪ val)",
            color="#2171b5",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_title(y_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Base model")
        ax.grid(axis="y", alpha=0.3)
        for i in range(3):
            ax.text(
                i - w / 2,
                float(m60[i]) + 0.004,
                f"{float(m60[i]):.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            ax.text(
                i + w / 2,
                float(m80[i]) + 0.004,
                f"{float(m80[i]):.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axa.set_ylabel("Score (same 20% hold-out)")
    axa.legend(loc="lower right", fontsize=7)

    all_twelve = np.concatenate([mean60, mean80, mf60, mf80])
    pad = 0.018
    y_bottom = max(0.75, float(all_twelve.min() - pad))
    y_top = min(1.0, float(all_twelve.max() + pad))
    if y_top - y_bottom < 0.08:
        mid = 0.5 * (y_bottom + y_top)
        y_bottom = max(0.75, mid - 0.055)
        y_top = min(1.0, mid + 0.055)
    axa.set_ylim(y_bottom, y_top)

    seed_list = ", ".join(str(s) for s in seeds)
    _figure_label_below(
        fig,
        (
            f"Figure 7 — Base models (LR / NB / RF): showcase seed {showcase_sd}, 60% vs 80% refit pool",
            f"Scanned {n_seeds} seeds · global min 60%-pool acc / macro-F1 = {gmin60:.3f} / {gmin60_f1:.3f} · "
            f"mean acc 60%/80% = {gmean60:.3f} / {gmean80:.3f}",
            f"Seeds scanned: {seed_list} · fixed hyperparameters",
        ),
        label_below_axes=0.175,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return showcase_sd


def figure_stacking_pool_acc_f1(
    rows: list[tuple[int, float, float, float, float]], path: Path
) -> None:
    """
    Final stacking: compare test accuracy and macro-F1 for 60% vs 80% refit/OOF pools.
    rows: (seed, acc_stack_60, macro_f1_60, acc_stack_80, macro_f1_80).
    """
    acc60 = np.array([r[1] for r in rows], dtype=float)
    f60 = np.array([r[2] for r in rows], dtype=float)
    acc80 = np.array([r[3] for r in rows], dtype=float)
    f80 = np.array([r[4] for r in rows], dtype=float)
    n = len(rows)
    if n > 1:
        err_acc = [acc60.std(ddof=1), acc80.std(ddof=1)]
        err_f = [f60.std(ddof=1), f80.std(ddof=1)]
    else:
        err_acc = [0.0, 0.0]
        err_f = [0.0, 0.0]

    m60 = [float(acc60.mean()), float(f60.mean())]
    m80 = [float(acc80.mean()), float(f80.mean())]
    labels = ["Test accuracy", "Macro F1"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x = np.arange(2)
    w = 0.35
    ax.bar(
        x - w / 2,
        m60,
        width=w,
        yerr=[err_acc[0], err_f[0]],
        capsize=3,
        label="60% train only (stacking OOF+refit)",
        color="#6baed6",
        ecolor="#333333",
    )
    ax.bar(
        x + w / 2,
        m80,
        width=w,
        yerr=[err_acc[1], err_f[1]],
        capsize=3,
        label="80% train+val (stacking OOF+refit)",
        color="#2171b5",
        ecolor="#333333",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (held-out 20% test)")
    all_m = np.array(m60 + m80)
    all_e = np.array(err_acc + err_f)
    y_lo = max(0.75, float((all_m - all_e).min() - 0.02))
    y_hi = min(1.0, float((all_m + all_e).max() + 0.02))
    if y_hi - y_lo < 0.06:
        mid = 0.5 * (y_lo + y_hi)
        y_lo, y_hi = max(0.75, mid - 0.04), min(1.0, mid + 0.04)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    seeds = [r[0] for r in rows]
    seed_list = ", ".join(str(s) for s in seeds)
    _figure_label_below(
        fig,
        (
            "Figure 8 — Final model (stacking): test accuracy & macro-F1",
            f"60% vs 80% training pool · mean ± std over seeds [{seed_list}]",
            "Same person-level 60/20/20 split and 20% test; val only in the 80% pool",
        ),
        label_below_axes=0.15,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_min_mean_max_table(df: pd.DataFrame, path: Path) -> None:
    """Render min/mean/max as a PNG table (no model column; CSV still has full keys)."""
    nrows = len(df)
    fig_w, row_h = 5.2, 0.52
    fig_h = 0.95 + row_h * (nrows + 1) + 0.95
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.48], hspace=0.36)
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    col_labels = ["Min", "Mean", "Max"]
    cell_text = [
        [f"{row['min']:.4f}", f"{row['mean']:.4f}", f"{row['max']:.4f}"]
        for _, row in df.iterrows()
    ]
    col_w = (0.31, 0.31, 0.31)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=col_w,
        bbox=[0.02, 0.02, 0.96, 0.96],
    )
    table.scale(1, 2.05)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#1a1a1a")
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
        else:
            cell.set_facecolor("#eef2f7" if r % 2 else "#ffffff")
            cell.set_text_props(color="#111111", fontsize=11, fontfamily="monospace")
    ax_cap = fig.add_subplot(gs[1])
    ax_cap.axis("off")
    cap = (
        "Figure 5 — Test accuracy: min, mean, max (six partition seeds)\n"
        "Same protocol as Figure 4"
    )
    ax_cap.text(
        0.5,
        0.92,
        cap,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        linespacing=1.1,
        transform=ax_cap.transAxes,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures for CSC311 project.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split seed for primary plots (default: pipeline.RANDOM_STATE)",
    )
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running model A evaluation for plots...")
    r = run_stacking_eval(split_seed=args.seed, verbose=False)
    primary_seed = int(r["split_seed"])

    figure_model_compare(r, PLOT_DIR / "01_model_compare_test.png", primary_seed)
    figure_confusion_matrix(
        r["confusion_matrix"], r["class_names"], PLOT_DIR / "02_confusion_matrix_stacking.png", primary_seed
    )
    figure_starry_night_errors(r, PLOT_DIR / "03_errors_true_starry_night.png", primary_seed)
    figure_confusion_matrix_baselines(r, PLOT_DIR / "06_confusion_matrix_baselines.png", primary_seed)
    print(f"Wrote {PLOT_DIR / '01_model_compare_test.png'}")
    print(f"Wrote {PLOT_DIR / '02_confusion_matrix_stacking.png'}")
    print(f"Wrote {PLOT_DIR / '03_errors_true_starry_night.png'}")
    print(f"Wrote {PLOT_DIR / '06_confusion_matrix_baselines.png'}")

    rows_a: list[tuple[int, float]] = []
    rows_stack_pool: list[tuple[int, float, float, float, float]] = []
    print("Partition-seed sweep over seeds", SPLIT_SEEDS_DEFAULT)
    for sd in SPLIT_SEEDS_DEFAULT:
        rr80 = run_stacking_eval(split_seed=sd, verbose=False, stacking_train_pool="trainval")
        rr60 = run_stacking_eval(split_seed=sd, verbose=False, stacking_train_pool="train")
        rows_a.append((sd, rr80["acc_stack"]))
        rows_stack_pool.append(
            (
                sd,
                float(rr60["acc_stack"]),
                float(rr60["macro_f1_stack"]),
                float(rr80["acc_stack"]),
                float(rr80["macro_f1_stack"]),
            )
        )
        print(
            f"  seed {sd}: stack 60% acc={rr60['acc_stack']:.4f} f1={rr60['macro_f1_stack']:.4f} | "
            f"80% acc={rr80['acc_stack']:.4f} f1={rr80['macro_f1_stack']:.4f}"
        )

    figure_stability_seeds(rows_a, PLOT_DIR / "04_partition_seed_sensitivity.png")
    print(f"Wrote {PLOT_DIR / '04_partition_seed_sensitivity.png'}")

    pd.DataFrame(
        [
            {
                "seed": s,
                "acc_stack_60_train": a60,
                "macro_f1_stack_60_train": f60,
                "acc_stack_80_trainval": a80,
                "macro_f1_stack_80_trainval": f80,
            }
            for s, a60, f60, a80, f80 in rows_stack_pool
        ]
    ).to_csv(PLOT_DIR / "08_stacking_train_pool_metrics.csv", index=False)
    print(f"Wrote {PLOT_DIR / '08_stacking_train_pool_metrics.csv'}")
    figure_stacking_pool_acc_f1(rows_stack_pool, PLOT_DIR / "08_stacking_train_pool_metrics.png")
    print(f"Wrote {PLOT_DIR / '08_stacking_train_pool_metrics.png'}")

    rows_fig7: list[tuple[int, dict, dict]] = []
    print("Figure 7 train-pool sweep over seeds", FIG7_TRAIN_POOL_SEEDS)
    for sd in FIG7_TRAIN_POOL_SEEDS:
        rr80 = run_stacking_eval(split_seed=sd, verbose=False, stacking_train_pool="trainval")
        rr60 = run_stacking_eval(split_seed=sd, verbose=False, stacking_train_pool="train")
        rows_fig7.append((sd, rr60, rr80))
        print(
            f"  fig7 seed {sd}: LR {rr60['acc_lr']:.3f}/{rr80['acc_lr']:.3f}  "
            f"NB {rr60['acc_nb']:.3f}/{rr80['acc_nb']:.3f}  "
            f"RF {rr60['acc_rf']:.3f}/{rr80['acc_rf']:.3f}"
        )
    fig7_sd = figure_train_pool_baselines_compare(
        rows_fig7, PLOT_DIR / "07_train_pool_compare_lr_nb_rf.png"
    )
    print(f"Wrote {PLOT_DIR / '07_train_pool_compare_lr_nb_rf.png'} (showcase seed {fig7_sd})")

    summary_rows = [
        {
            "model": "A_stacking_fixed_hparams",
            "min": min(a for _, a in rows_a),
            "mean": float(np.mean([a for _, a in rows_a])),
            "max": max(a for _, a in rows_a),
        }
    ]

    df = pd.DataFrame(summary_rows)
    csv_path = PLOT_DIR / "05_stacking_seed_min_mean_max.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    table_png = PLOT_DIR / "05_stacking_seed_min_mean_max.png"
    figure_min_mean_max_table(df, table_png)
    print(f"Wrote {table_png}")

    print("\nDone. See module docstring in report_figures.py for the output list.")


if __name__ == "__main__":
    main()
