# CSC311 Project — `cheng2` branch (development)

This branch keeps the **stacking** line of work: LR + NB/CNB + RF with a **meta logistic regression** on out-of-fold base probabilities (`stacking_ensemble.py`). Other directions (e.g. softmax-only) were removed to avoid clutter.

## Validation / multiseed scripts (why we stuck with stacking)

| Script | Purpose |
|--------|---------|
| **`stacking_seed_eval_fast.py`** | Six split seeds; reports 60/20/20 vs **80/20** stacking accuracy (edit `META_C` / `RF_*` at top to reproduce a setting). |
| **`stacking_min_priority_eval.py`** | Ten seeds; compares named configs (baseline vs RF/meta tweaks) ranked by **worst-seed accuracy first** (then p25, mean). |

Run from repo root with `training_data.csv` present:

```bash
python stacking_seed_eval_fast.py
python stacking_min_priority_eval.py
```

## Core library files

- **`stacking_ensemble.py`** — Full training script with Stage‑1 CV for LR/NB/RF grid and OOF stacking (research / class demo).
- **`pipeline.py`**, **`naive_bayes.py`**, **`data_exploration.py`** — Shared preprocessing, NB features, EDA.

## Submission-ready snapshot

The cleaned **MarkUs** layout (fixed hyperparameters, `pred.py`, `export_model.py`) lives on the **`final`** branch:

```bash
git checkout final
```

---

*Original exploration lived here; `final` is what we submit.*
