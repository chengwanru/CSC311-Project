"""
Generate figures and CSV tables for the CSC311 report (Results + Appendix).

Dependencies: numpy, pandas, scikit-learn, matplotlib
(`pip install -r requirements-figures.txt`).

PNG plots and CSV tables are written to ``./plots/`` (created automatically).

Usage (repository root, ``training_data.csv`` present)::

    python report_figures.py              # default partition seed 42 + six-seed sweep
    python report_figures.py --appendix   # extra bar chart over alternative setups (slow)

**Outputs (rubric mapping)**

``plots/01_model_compare_test.png``
    Bar chart: test-set accuracy and macro-F1 for LR, NB, RF, majority vote, stacking.

``plots/02_confusion_matrix_stacking.png``
    Confusion matrix for stacked classifier on the 20% test set.

``plots/03_errors_true_starry_night.png``
    True class Starry Night: distribution of predicted labels on the test set.

``plots/04_partition_seed_sensitivity.png``
    Stacking test accuracy vs random seed for the train/test split (same 80/20 protocol).

``plots/05_appendix_model_compare_min_mean_max.csv``
    Min/mean/max table; ``--appendix`` adds extra rows and ``05_appendix_model_compare.png``.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stacking_ensemble import run_stacking_eval

SPLIT_SEEDS_DEFAULT = [1, 7, 13, 21, 42, 84]
PLOT_DIR = Path(__file__).resolve().parent / "plots"


def _figure_label_below(fig, lines: tuple[str, ...]) -> None:
    """Multi-line figure title under the axes (reserve bottom margin first)."""
    fig.tight_layout(rect=[0, 0.20, 1, 0.98])
    fig.text(
        0.5,
        0.02,
        "\n".join(lines),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        transform=fig.transFigure,
        linespacing=1.35,
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
            "Figure 1 — Test set: accuracy vs macro-F1",
            "LR, NB, RF, majority vote, stacked meta-classifier",
            "Split: 80% respondents train+val, 20% test set",
            f"Seed {split_seed}",
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def figure_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: Path, split_seed: int
) -> None:
    short = _short_labels(class_names)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short,
        yticklabels=short,
        ylabel="True class",
        xlabel="Predicted class",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
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
            "Figure 2 — Confusion matrix (stacked classifier, test set)",
            "Rows: true class, columns: predicted class (counts)",
            "Split: 80% respondents train+val, 20% test set",
            f"Seed {split_seed}",
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
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
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(short, counts, color=["#4a90d9" if i != sn_i else "#e94b3c" for i in range(len(names))])
    ax.set_ylabel("Test row count")
    ax.grid(axis="y", alpha=0.3)
    _figure_label_below(
        fig,
        (
            "Figure 3 — True class: The Starry Night (test set rows only)",
            "Counts of predicted class by the stacked model",
            "Same split as Figures 1–2",
            f"Seed {split_seed}",
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def figure_stability_seeds(rows: list[tuple[int, float]], path: Path) -> None:
    seeds, accs = zip(*rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(seeds, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Random seed (80/20 train+val vs test, same protocol each run)")
    ax.set_ylabel("Stacking accuracy on 20% test set")
    ax.grid(True, alpha=0.3)
    m = min(accs)
    ax.axhline(m, color="gray", linestyle="--", alpha=0.7, label=f"min={m:.4f}")
    ax.axhline(np.mean(accs), color="green", linestyle=":", alpha=0.8, label=f"mean={np.mean(accs):.4f}")
    ax.legend(loc="lower right")
    seed_list = ", ".join(str(s) for s in seeds)
    _figure_label_below(
        fig,
        (
            "Figure 4 — Stacking test accuracy vs random seed",
            f"Seeds: {seed_list}",
            "Hyperparameters fixed; only the random train/test split changes",
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def figure_appendix_bar(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary))
    w = 0.25
    ax.bar(x - w, summary["min"], width=w, label="min")
    ax.bar(x, summary["mean"], width=w, label="mean")
    ax.bar(x + w, summary["max"], width=w, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=15, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    n = len(summary)
    mid = (
        "Six random seeds per configuration (same protocol as Figure 4)"
        if n > 1
        else "Default script: one stacking configuration (--appendix adds more)"
    )
    _figure_label_below(
        fig,
        (
            "Figure 5 — Test accuracy: min, mean, max over random seeds",
            mid,
            f"{n} configuration(s)",
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _load_appendix_module(filename: str, unique_tag: str):
    root = Path(__file__).resolve().parent
    p = root / "appendix_code" / filename
    spec = importlib.util.spec_from_file_location(f"appendix_{unique_tag}", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def eval_appendix_B_or_C(which: str, split_seed: int) -> float:
    import pipeline

    pipeline.RANDOM_STATE = split_seed
    tag = f"{which}_{split_seed}"
    if which == "B":
        mod = _load_appendix_module("stacking_multiseed.py", tag)
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            mod.main()
        text = f.getvalue()
        for line in text.splitlines():
            if "Multi-seed stacking" in line and "acc=" in line:
                import re

                m = re.search(r"acc=([\d.]+)", line)
                if m:
                    return float(m.group(1))
        raise RuntimeError("Could not parse B accuracy from output")
    if which == "C":
        mod = _load_appendix_module("stacking_ensemble_multiRF.py", tag)
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            mod.main()
        text = f.getvalue()
        for line in text.splitlines():
            if "Stacking" in line and "9-dim" in line and "acc=" in line:
                import re

                m = re.search(r"acc=([\d.]+)", line)
                if m:
                    return float(m.group(1))
        raise RuntimeError("Could not parse C accuracy from output")
    raise ValueError(which)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures for CSC311 project.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split seed for primary plots (default: pipeline.RANDOM_STATE)",
    )
    parser.add_argument(
        "--appendix",
        action="store_true",
        help="Also evaluate appendix models B and C over SPLIT_SEEDS (slow).",
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
    print(f"Wrote {PLOT_DIR / '01_model_compare_test.png'}")
    print(f"Wrote {PLOT_DIR / '02_confusion_matrix_stacking.png'}")
    print(f"Wrote {PLOT_DIR / '03_errors_true_starry_night.png'}")

    rows_a: list[tuple[int, float]] = []
    print("Partition-seed sweep over seeds", SPLIT_SEEDS_DEFAULT)
    for sd in SPLIT_SEEDS_DEFAULT:
        rr = run_stacking_eval(split_seed=sd, verbose=False)
        rows_a.append((sd, rr["acc_stack"]))
        print(f"  seed {sd}: acc_stack={rr['acc_stack']:.4f}")

    figure_stability_seeds(rows_a, PLOT_DIR / "04_partition_seed_sensitivity.png")
    print(f"Wrote {PLOT_DIR / '04_partition_seed_sensitivity.png'}")

    summary_rows = [
        {
            "model": "A_stacking_fixed_hparams",
            "min": min(a for _, a in rows_a),
            "mean": float(np.mean([a for _, a in rows_a])),
            "max": max(a for _, a in rows_a),
        }
    ]

    if args.appendix:
        for label, key in [("B_multiseed_meta45", "B"), ("C_RF_avg_meta9", "C")]:
            accs = []
            for sd in SPLIT_SEEDS_DEFAULT:
                print(f"Appendix {key} split_seed={sd} ...")
                accs.append(eval_appendix_B_or_C(key, sd))
                print(f"  acc={accs[-1]:.4f}")
            summary_rows.append(
                {
                    "model": label,
                    "min": min(accs),
                    "mean": float(np.mean(accs)),
                    "max": max(accs),
                }
            )

    df = pd.DataFrame(summary_rows)
    csv_path = PLOT_DIR / "05_appendix_model_compare_min_mean_max.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    if args.appendix and len(summary_rows) > 1:
        figure_appendix_bar(df, PLOT_DIR / "05_appendix_model_compare.png")
        print(f"Wrote {PLOT_DIR / '05_appendix_model_compare.png'}")

    print("\nDone. See module docstring in report_figures.py for output list / rubric mapping.")


if __name__ == "__main__":
    main()
