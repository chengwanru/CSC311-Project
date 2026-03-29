# CSC311 Project — `final` branch

This branch matches the **course submission layout** (Winter 2026 CSC311 project instructions): MarkUs **prediction** (`pred.py` + small artifacts), **report** (`report.pdf` + `code.zip` evidence).

## Final model choice (**model A**)

The **submitted** classifier is **stacking model A**: fixed hyperparameters, three base models (LR, NB/CNB blend, RF), **9-dimensional** OOF meta-features, meta logistic regression **C = 0.5** — implemented in `stacking_ensemble.py` (evaluation) and `export_model.py` (full-data train for `pred.py`).

**Appendix-only code (models B & C)** lives in `appendix_code/` (see `appendix_code/README.md`). Include that folder in **code.zip** for the report; it is **not** loaded by `pred.py`.

## Our approach

We use **stacking**: three base classifiers each output class probabilities; a **meta-model** (multinomial logistic regression) learns how to combine those probabilities. Final prediction is the meta-model’s argmax over three painting classes.

**Base models**

1. **Logistic regression** on a dense feature vector: numeric / Likert / price (scaled), multi-hot room–companion–season, and **TF–IDF** over description + food + soundtrack text (`pipeline.fit_state` / `transform_df`). Vocabulary is fit on the training portion only to avoid leakage.
2. **Custom Naive Bayes**: multinomial block + Gaussian block on continuous numeric features, with a **CNB** branch; we blend log-probabilities (here effectively **NB-only**, blend weight 1.0). Features come from `naive_bayes.build_features` (different tokenisation than LR’s TF–IDF).
3. **Random forest** on the **same** dense matrix as the LR (not the NB matrix).

**How stacking is trained**

- Data: `training_data.csv` after `pipeline.clean()`, rows aligned by `(unique_id, Painting)`.
- **Person-level split**: 60% train / 20% val / 20% test (`regular_split`). LR preprocessing state is fit on **train only**; val and test are transformed with that state.
- **Meta-training data**: on the **train+val pool (80%)**, we run **5-fold, person-level OOF**: each fold trains the three bases on four folds and records **probabilities** on the fifth. Those nine values per row (3 models × 3 classes) are the meta-features.
- **Meta-classifier**: logistic regression on those OOF features (**C = 0.5**). Then we **refit** all three bases on the full 80% and evaluate on the 20% test set by stacking their test probabilities through the trained meta-model.

**Fixed hyperparameters** (chosen from multiseed checks; see `stacking_ensemble.py`): LR `C = 100`, NB `α = 0.9`, NB/CNB blend `1.0`, RF `n_estimators = 200`, `max_depth = None`, `min_samples_leaf = 1`, meta `C = 0.5`.

**Submission model** (`export_model.py`): same recipe, but **no held-out test** — OOF on **all** cleaned rows, then refit bases on **all** rows, and export weights for `pred.py`.

---

## What each file is for


| File                   | Role                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `stacking_ensemble.py` | Local **evaluation** for **model A**: 60/20/20 split, OOF stacking on 80%, metrics on 20% test. Exposes `run_stacking_eval()` for scripts. Uses **sklearn**.               |
| `export_model.py`      | **Train on full** `training_data.csv` and write `model_state.json` + `model_weights.npz` for MarkUs prediction. Uses **sklearn**.                                        |
| `pred.py`              | **MarkUs prediction entry**: `predict_all(csv_path)` → list of painting name strings. **Only** stdlib, **numpy**, **pandas** — loads the two artifact files, no sklearn. |
| `report_figures.py`    | Optional: builds **`plots/`** and CSV tables for the report (`pip install -r requirements-figures.txt`).                                                                |
| `appendix_code/`       | **Models B & C** + `stacking_experiments.py` (multiseed meta-C sweep). Not used by `pred.py`; zip for report evidence.                                                    |
| `pipeline.py`          | Cleaning, person-level splits, TF–IDF vocab/IDF, `fit_state` / `transform_df` for LR/RF features.                                                                        |
| `naive_bayes.py`       | NB/CNB feature matrix, training helpers, and the same NB logic the export script uses.                                                                                   |
| `data_exploration.py`  | Exploratory analysis script; include in **code.zip** if you use it as report evidence.                                                                                   |
| `training_data.csv`    | Course training data (keep in repo / push as your team agrees).                                                                                                          |
| `LICENSE`              | License for the repo.                                                                                                                                                    |
| `.gitignore`           | Ignores caches, venv, and generated `model_state.json` / `model_weights.npz` (regenerate before submit).                                                                 |


---

## What to submit (per course instructions)

**Prediction assignment (MarkUs)**

- **`pred.py`** — Python **3.10+**; imports restricted to **stdlib, numpy, pandas** only; must define **`predict_all(csv_path)`** returning predictions; **no networking**; should run ~60 predictions within **~1 minute** with reasonable memory.
- **`model_state.json`** and **`model_weights.npz`** — generate with `python export_model.py`; combined size **≤ 10 MB**.

**Report assignment (MarkUs)**

- **`report.pdf`**
- **`code.zip`** — all `**.py`** / `**.ipynb**` used to develop the final model. **Exclude** any `/data` folder per instructions. The zip is **evidence only** (need not be runnable on the TA machine); include **`appendix_code/`** if you discuss models B/C in the appendix.

Suggested **code.zip** contents: `pred.py`, `export_model.py`, `stacking_ensemble.py`, `pipeline.py`, `naive_bayes.py`, `data_exploration.py`, `report_figures.py`, `appendix_code/` (entire folder), `requirements-figures.txt`.

---

## Report figures (Results / Appendix)

Install optional dependency: `pip install -r requirements-figures.txt`

```bash
# Default split seed + stability over 6 person-level split seeds (model A)
python report_figures.py

# Also evaluate appendix B & C over the same seeds (slow)
python report_figures.py --appendix
```

Outputs go to **`plots/`** — see **`plots/README.md`** and the **module docstring** in `report_figures.py`. Each PNG uses a short **multi-line title** (no long footnote caption, so aspect ratio stays natural).

---

## Quick commands

```bash
# Evaluate stacking with a held-out 20% test set
python stacking_ensemble.py

# Build artifacts for pred.py (full-data training)
python export_model.py

# Smoke test (expects artifacts next to pred.py)
python pred.py training_data.csv
```

---

## Branch note

- **`cheng2`** — Original development code (full history / experiments).
- **`final`** — Submission-ready version: cleaned layout, final stacking setup, `pred.py` + export pipeline, and this README.

Use `git checkout final` when preparing MarkUs files; use `cheng2` if you need the older tree.
