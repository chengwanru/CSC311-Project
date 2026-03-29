"""
Generate figures and CSV tables for the CSC311 report (Results + Appendix).

Dependencies: numpy, pandas, scikit-learn, matplotlib
(`pip install -r requirements-figures.txt`).

Figures are written to ./figures/ (created automatically).

Usage (repository root, ``training_data.csv`` present)::

    python report_figures.py              # seed 42 plots + 6-seed stability (model A)
    python report_figures.py --appendix   # also B/C over 6 seeds (slow)

**Outputs (rubric mapping)**

``01_model_compare_test.png``
    Bar chart: test accuracy + macro-F1 for LR, NB, RF, majority vote, stacking (model A).

``02_confusion_matrix_stacking.png``
    Confusion matrix heatmap (stacking, held-out 20% test, default split seed 42).

``03_errors_true_starry_night.png``
    When true class is *The Starry Night*, bar chart of predicted labels.

``04_stability_model_A_split_seeds.png``
    Test accuracy vs person-level split seed (6 seeds) for model A.

``05_appendix_model_compare_min_mean_max.csv``
    Min / mean / max test accuracy. With ``--appendix``, includes models B and C;
    also writes ``05_appendix_model_compare.png``. Without ``--appendix``, model A only.
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
FIG_DIR = Path(__file__).resolve().parent / "figures"


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


def figure_model_compare(r: dict, path: Path) -> None:
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
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, accs, width=w, label="Accuracy")
    ax.bar(x + w / 2, f1s, width=w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test: base models vs stacking (model A)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    short = _short_labels(class_names)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short,
        yticklabels=short,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion matrix — stacking (test set)",
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
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_starry_night_errors(r: dict, path: Path) -> None:
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
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(short, counts, color=["#4a90d9" if i != sn_i else "#e94b3c" for i in range(len(names))])
    ax.set_ylabel("Count (test rows)")
    ax.set_title('Predicted label when true class is "The Starry Night"')
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_stability_seeds(rows: list[tuple[int, float]], path: Path) -> None:
    seeds, accs = zip(*rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(seeds, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Person-level split seed (regular_split)")
    ax.set_ylabel("Test accuracy (stacking, model A)")
    ax.set_title("Stability across split seeds")
    ax.grid(True, alpha=0.3)
    m, M = min(accs), max(accs)
    ax.axhline(m, color="gray", linestyle="--", alpha=0.7, label=f"min={m:.4f}")
    ax.axhline(np.mean(accs), color="green", linestyle=":", alpha=0.8, label=f"mean={np.mean(accs):.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_appendix_bar(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary))
    w = 0.25
    ax.bar(x - w, summary["min"], width=w, label="min")
    ax.bar(x, summary["mean"], width=w, label="mean")
    ax.bar(x + w, summary["max"], width=w, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"])
    ax.set_ylabel("Test accuracy")
    ax.set_title("Appendix: min / mean / max over person-level split seeds")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Running model A evaluation for plots...")
    r = run_stacking_eval(split_seed=args.seed, verbose=False)

    figure_model_compare(r, FIG_DIR / "01_model_compare_test.png")
    figure_confusion_matrix(r["confusion_matrix"], r["class_names"], FIG_DIR / "02_confusion_matrix_stacking.png")
    figure_starry_night_errors(r, FIG_DIR / "03_errors_true_starry_night.png")
    print(f"Wrote {FIG_DIR / '01_model_compare_test.png'}")
    print(f"Wrote {FIG_DIR / '02_confusion_matrix_stacking.png'}")
    print(f"Wrote {FIG_DIR / '03_errors_true_starry_night.png'}")

    rows_a: list[tuple[int, float]] = []
    print("Stability sweep model A over seeds", SPLIT_SEEDS_DEFAULT)
    for sd in SPLIT_SEEDS_DEFAULT:
        rr = run_stacking_eval(split_seed=sd, verbose=False)
        rows_a.append((sd, rr["acc_stack"]))
        print(f"  seed {sd}: acc_stack={rr['acc_stack']:.4f}")

    figure_stability_seeds(rows_a, FIG_DIR / "04_stability_model_A_split_seeds.png")
    print(f"Wrote {FIG_DIR / '04_stability_model_A_split_seeds.png'}")

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
    csv_path = FIG_DIR / "05_appendix_model_compare_min_mean_max.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    if args.appendix and len(summary_rows) > 1:
        figure_appendix_bar(df, FIG_DIR / "05_appendix_model_compare.png")
        print(f"Wrote {FIG_DIR / '05_appendix_model_compare.png'}")

    print("\nDone. See module docstring in report_figures.py for output list / rubric mapping.")


if __name__ == "__main__":
    main()
