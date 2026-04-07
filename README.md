# Painting prediction — stacking ensemble

**Multiclass classification** on survey data: predict which of three famous paintings a participant preferred (*The Starry Night*, *The Water Lily Pond*, *The Persistence of Memory*).  
Course project for **CSC311 — Introduction to Machine Learning**, University of Toronto, Winter 2026.

---

## Overview

- **Stacking ensemble** — logistic regression, **custom Naive Bayes** (multinomial + Gaussian + Complement NB branch), and random forest; **meta-learner** is multinomial logistic regression on **9** out-of-fold probability features (3 models × 3 classes).
- **Leakage-aware NLP** — TF–IDF vocabulary and scaling statistics are fit **only on the training split**; splits are **person-level** (same respondent stays in one fold).
- **Two-stage workflow** — sklearn for training/export; **`pred.py` inference uses only the Python standard library, NumPy, and Pandas** (weights loaded from exported JSON + NPZ).
- **Figures & tables** — `report_figures.py` regenerates evaluation plots and CSV summaries under `plots/`.

---

## Tech stack

| Area | Details |
|------|---------|
| Language | Python **3.10+** |
| Training / eval | NumPy, Pandas, **scikit-learn** |
| Inference | stdlib + NumPy + Pandas (no sklearn at predict time) |
| Figures | Matplotlib (+ scikit-learn for evaluations inside the script) |

---

## Repository layout

```
├── pipeline.py           # Cleaning, person-level splits, TF–IDF, LR/RF features
├── naive_bayes.py        # NB/CNB feature construction and training helpers
├── stacking_ensemble.py    # 60/20/20 eval, 5-fold OOF stacking, test metrics
├── export_model.py       # Full-data train → model_state.json + model_weights.npz
├── pred.py               # predict_all(csv_path) — batch inference from CSV
├── report_figures.py     # Regenerates plots/ (see requirements-figures.txt)
├── training_data.csv     # Labeled survey responses (~1.8k rows), course-provided
├── plots/                # Pre-generated figures + CSV summaries
├── requirements.txt      # Core ML dependencies
└── requirements-figures.txt
```

---

## Data

**`training_data.csv`** is the course-provided labeled dataset (included in the repo). Each row links a participant’s ratings, free-text answers, and Likert-style fields to one of the three painting labels. Feature names and cleaning rules are implemented in `pipeline.py` and `pred.py`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional — only if you want to rebuild figures:

```bash
pip install -r requirements-figures.txt
```

---

## Usage

### Evaluate stacking (held-out 20% test, person-level split)

```bash
python stacking_ensemble.py
```

Uses fixed hyperparameters (see `stacking_ensemble.py`): e.g. LR `C=100`, NB `α=0.9`, RF `200` trees, meta logistic `C=0.5`. Quick multiseed checks in the module docstring report **~0.92–0.94** test accuracy depending on partition seed.

### Export weights for inference

```bash
python export_model.py
```

Writes **`model_state.json`** and **`model_weights.npz`** next to `pred.py` (gitignored by default — regenerate after clone).

### Run inference

```bash
python pred.py path/to/input.csv
```

`pred.py` defines **`predict_all(csv_path)`** → `list[str]` of painting names. Input columns should match the training schema (missing columns are handled conservatively for test-style CSVs).

### Regenerate report figures

```bash
python report_figures.py
```

Outputs are described in **`plots/README.md`** (model comparison bars, confusion matrices, partition-seed stability, train-pool ablations).

---

## Method (short)

1. **Preprocess** — `pipeline.clean()` and align rows by `(unique_id, Painting)`.
2. **Split** — **60% train / 20% validation / 20% test** by person; LR preprocessing state fit on **train only**.
3. **Base models** — LR and RF share the same dense design matrix; NB uses its own discrete/continuous feature blocks.
4. **Meta-training** — On the **80% train+val pool**, **5-fold person-level OOF** produces class probabilities from each base model; **9** stacked probabilities per row feed a **multinomial logistic regression** meta-classifier.
5. **Evaluation** — Refit bases on full 80%, stack on the 20% test set. **Export** retrains on **all** cleaned rows and saves weights for `pred.py`.
