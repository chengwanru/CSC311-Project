"""
Generate figures and CSV tables for the CSC311 report (Results + Appendix).

Dependencies: numpy, pandas, scikit-learn, matplotlib
(`pip install -r requirements-figures.txt`).

PNG plots and CSV tables are written to ``./plots/`` (created automatically).

Usage (repository root, ``training_data.csv`` present)::

    python report_figures.py              # seed 42 plots + 6-seed stability (model A)
    python report_figures.py --appendix   # also B/C over 6 seeds (slow)

**Outputs (rubric mapping)**

``plots/01_model_compare_test.png``
    Bar chart: test accuracy + macro-F1 for LR, NB, RF, majority vote, stacking (model A).

``plots/02_confusion_matrix_stacking.png``
    Confusion matrix heatmap (stacking, held-out 20% test, default split seed 42).

``plots/03_errors_true_starry_night.png``
    When true class is *The Starry Night*, bar chart of predicted labels.

``plots/04_stability_model_A_split_seeds.png``
    Test accuracy vs person-level split seed (6 seeds) for model A.

``plots/05_appendix_model_compare_min_mean_max.csv``
    Min / mean / max test accuracy. With ``--appendix``, includes models B and C;
    also writes ``plots/05_appendix_model_compare.png``. Without ``--appendix``, model A only.
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


def _finish_figure(fig, *, suptitle: str, caption: str) -> None:
    """Bold figure title (what plot) + caption line (protocol / how to read)."""
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])


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
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, accs, width=w, label="Accuracy")
    ax.bar(x + w / 2, f1s, width=w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Metrics on held-out 20% test (same split as stacking evaluation)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _finish_figure(
        fig,
        suptitle="Figure 1 — Model comparison: accuracy and macro-F1 (final model A)",
        caption=(
            f"Caption: Bar chart comparing five predictors on the test fold only — "
            f"logistic regression, custom NB/CNB, random forest, majority vote of the three bases, "
            f"and stacked meta-logistic regression (9 OOF probability features). "
            f"Person-level 60/20/20 split via pipeline.regular_split; split_seed={split_seed}."
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: Path, split_seed: int
) -> None:
    short = _short_labels(class_names)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short,
        yticklabels=short,
        ylabel="True class",
        xlabel="Predicted class",
        title="Stacking classifier — row = true label, column = prediction",
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
    _finish_figure(
        fig,
        suptitle="Figure 2 — Confusion matrix for stacked meta-model (test set)",
        caption=(
            f"Caption: Counts of (true, predicted) pairs for the final stacking model on held-out test rows; "
            f"diagonal = correct. Off-diagonals show which classes are confused (e.g. Starry Night vs others). "
            f"split_seed={split_seed}, person-level split."
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(short, counts, color=["#4a90d9" if i != sn_i else "#e94b3c" for i in range(len(names))])
    ax.set_ylabel("Number of test rows")
    ax.set_title('Subset: ground truth = "The Starry Night" only')
    ax.grid(axis="y", alpha=0.3)
    _finish_figure(
        fig,
        suptitle="Figure 3 — Error analysis: predictions when true class is Starry Night",
        caption=(
            f"Caption: Among test rows whose true painting is *The Starry Night*, bar height shows how often "
            f"the stacking model predicted each class. Highlights the most confused target class. "
            f"split_seed={split_seed}."
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_stability_seeds(rows: list[tuple[int, float]], path: Path) -> None:
    seeds, accs = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(seeds, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Person-level split seed (pipeline.regular_split)")
    ax.set_ylabel("Test accuracy (stacking, model A)")
    ax.set_title("Same hyperparameters; only the train/val/test person split changes")
    ax.grid(True, alpha=0.3)
    m = min(accs)
    ax.axhline(m, color="gray", linestyle="--", alpha=0.7, label=f"min={m:.4f}")
    ax.axhline(np.mean(accs), color="green", linestyle=":", alpha=0.8, label=f"mean={np.mean(accs):.4f}")
    ax.legend(loc="lower right")
    seed_list = ", ".join(str(s) for s in seeds)
    _finish_figure(
        fig,
        suptitle="Figure 4 — Test accuracy vs split seed (stability of final model A)",
        caption=(
            f"Caption: Each point retrains the full stacking pipeline with a different random seed for "
            f"person-level 60/20/20 splitting; horizontal lines show min and mean test accuracy over seeds "
            f"[{seed_list}]. Supports reporting a performance range, not a single lucky split."
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_appendix_bar(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    w = 0.25
    ax.bar(x - w, summary["min"], width=w, label="min")
    ax.bar(x, summary["mean"], width=w, label="mean")
    ax.bar(x + w, summary["max"], width=w, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=15, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Same six split seeds for every model variant")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _finish_figure(
        fig,
        suptitle="Figure 5 — Appendix: min / mean / max test accuracy across split seeds",
        caption=(
            "Caption: Summary over the same person-level split seeds for final model A vs exploratory "
            "variants B (multiseed meta features) and C (RF probability averaging). "
            "Run report_figures.py --appendix to include B and C in this chart."
        ),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    print("Stability sweep model A over seeds", SPLIT_SEEDS_DEFAULT)
    for sd in SPLIT_SEEDS_DEFAULT:
        rr = run_stacking_eval(split_seed=sd, verbose=False)
        rows_a.append((sd, rr["acc_stack"]))
        print(f"  seed {sd}: acc_stack={rr['acc_stack']:.4f}")

    figure_stability_seeds(rows_a, PLOT_DIR / "04_stability_model_A_split_seeds.png")
    print(f"Wrote {PLOT_DIR / '04_stability_model_A_split_seeds.png'}")

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
