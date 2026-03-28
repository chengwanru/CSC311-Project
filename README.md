# CSC311 Project — `final` branch

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
| `stacking_ensemble.py` | Local **evaluation**: 60/20/20 split, OOF stacking on 80%, metrics on 20% test. Uses **sklearn**.                                                                        |
| `export_model.py`      | **Train on full** `training_data.csv` and write `model_state.json` + `model_weights.npz` for MarkUs prediction. Uses **sklearn**.                                        |
| `pred.py`              | **MarkUs prediction entry**: `predict_all(csv_path)` → list of painting name strings. **Only** stdlib, **numpy**, **pandas** — loads the two artifact files, no sklearn. |
| `pipeline.py`          | Cleaning, person-level splits, TF–IDF vocab/IDF, `fit_state` / `transform_df` for LR/RF features.                                                                        |
| `naive_bayes.py`       | NB/CNB feature matrix, training helpers, and the same NB logic the export script uses.                                                                                   |
| `data_exploration.py`  | Exploratory analysis script; include in **code.zip** if you use it as report evidence.                                                                                   |
| `training_data.csv`    | Course training data (keep in repo / push as your team agrees).                                                                                                          |
| `LICENSE`              | License for the repo.                                                                                                                                                    |
| `.gitignore`           | Ignores caches, venv, and generated `model_state.json` / `model_weights.npz` (regenerate before submit).                                                                 |


---

## What to submit

**Prediction assignment**

- `pred.py`
- `model_state.json` and `model_weights.npz` (run `python export_model.py` first; combined size must stay under **10 MB**)

**Report assignment**

- `report.pdf`
- `code.zip` — all `**.py`** (and any `**.ipynb**`) you used to **develop** the final model (e.g. the files above; **exclude** a `/data` folder if you have one, per instructions). This zip is evidence only; it does not need to be runnable on the TA’s machine.

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
