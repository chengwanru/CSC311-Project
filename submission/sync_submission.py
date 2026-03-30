#!/usr/bin/env python3
"""
Copy submission-ready files from the repo root into submission/* staging folders.

Run from repository root:
    python submission/sync_submission.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = Path(__file__).resolve().parent
PRED = SUB / "01_prediction_markus"
CODE = SUB / "02_report_code_zip"

PRED_FILES = ["pred.py", "model_state.json", "model_weights.npz"]

CODE_ROOT_FILES = [
    "pred.py",
    "export_model.py",
    "stacking_ensemble.py",
    "pipeline.py",
    "naive_bayes.py",
    "data_exploration.py",
    "report_figures.py",
    "requirements-figures.txt",
    "LICENSE",
]


def main() -> None:
    if not (ROOT / "pred.py").is_file():
        print("Error: run from repository root (pred.py not found).", file=sys.stderr)
        sys.exit(1)

    PRED.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)

    missing_art = []
    for name in PRED_FILES:
        src = ROOT / name
        if not src.is_file():
            if name != "pred.py":
                missing_art.append(name)
            continue
        shutil.copy2(src, PRED / name)
        print(f"  prediction: {name}")

    for name in CODE_ROOT_FILES:
        src = ROOT / name
        if not src.is_file():
            print(f"  skip (missing): {name}")
            continue
        shutil.copy2(src, CODE / name)
        print(f"  code_zip: {name}")

    appendix = ROOT / "appendix_code"
    if appendix.is_dir():
        dest = CODE / "appendix_code"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(appendix, dest)
        print("  code_zip: appendix_code/ (full tree)")
    else:
        print("  warning: appendix_code/ not found")

    plots_dir = ROOT / "plots"
    if plots_dir.is_dir():
        dest_plots = CODE / "plots"
        if dest_plots.exists():
            shutil.rmtree(dest_plots)
        dest_plots.mkdir(parents=True, exist_ok=True)
        for p in sorted(plots_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in (".png", ".csv", ".md"):
                shutil.copy2(p, dest_plots / p.name)
                print(f"  code_zip: plots/{p.name}")

    print("\nDone.")
    if missing_art:
        print(
            "\nMissing prediction artifacts — run from repo root:\n  python export_model.py\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
