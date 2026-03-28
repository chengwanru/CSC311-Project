# CSC311 Project — Final submission branch (`final`)

## What’s in this branch

- **`stacking_ensemble.py`** — Final **stacking** model (LR + NB/CNB + RF → meta logistic regression) with fixed hyperparameters chosen from multiseed evaluation. Uses a person-level **60/20/20** split and reports test metrics on the held-out 20%.
- **`export_model.py`** — Retrains the **same** stacking setup on **100%** of `training_data.csv` and writes **`model_state.json`** + **`model_weights.npz`** for deployment.
- **`pred.py`** — MarkUs **prediction script**: `predict_all(csv_path)` using **only** the Python standard library, **numpy**, and **pandas** (no sklearn). Requires the two model files from `export_model.py`.
- **`pipeline.py`**, **`naive_bayes.py`** — Shared preprocessing and NB feature pipeline.
- **`data_exploration.py`** — EDA script for **code.zip** / report evidence (optional to run).

## MarkUs checklist (from project instructions)

| Deliverable | Notes |
|-------------|--------|
| **pred.py** | Must define `predict_all(csv_path)`. Imports: stdlib, numpy, pandas only. |
| **Bundled files** | Submit `pred.py` + `model_state.json` + `model_weights.npz` (total &lt; 10 MB). |
| **code.zip** (report) | Include `.py` files used to **develop** the model (this repo’s `.py` files; exclude `/data` if present). |

## Quick commands

```bash
# Local evaluation (80% train+val OOF stacking, 20% test)
python stacking_ensemble.py

# Regenerate artifacts for pred.py (train on all rows in training_data.csv)
python export_model.py

# Smoke test predictions
python pred.py training_data.csv
```

## Branch note

- **`cheng2`** is unchanged: all cleanup and final layout live on **`final`** only.
