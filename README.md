# CSC311 Project — `cheng2` branch (stacking-ensemble lineage)

This branch preserves the **stacking** approach from the original `stacking-ensemble` work: **LR + NB/CNB + RF → meta logistic regression** on **5-fold person-level OOF** probabilities (`stacking_ensemble.py`). Softmax-only code was removed.

Later we tuned **meta-LR `C`** and **RF shape** to improve **worst-split (min)** test accuracy without sacrificing **mean** too much. All multiseed checks live in **one script** so you can add presets instead of new files.

---

## Single experiment driver

```bash
python stacking_experiments.py           # all presets in PRESETS
python stacking_experiments.py --preset C
python stacking_experiments.py --rank    # sort runs by min, then mean
```

Edit **`PRESETS`** in `stacking_experiments.py` (copy a row, change `id` / numbers) to try new combinations.

---

## Parameter meanings (what we varied)

| Symbol / field | Meaning |
|----------------|--------|
| **`meta_c`** | Sklearn `LogisticRegression(C=…)` for the **meta-model** (9 stacked base probs). **Smaller `C` ⇒ stronger L2** on meta weights ⇒ less overfit to OOF noise, often **higher min** across split seeds at some **mean** cost. |
| **`rf_n_est`** | Number of trees in **RandomForest** (base model 3). |
| **`rf_depth`** | `max_depth`; **`None`** = unlimited depth (more flexible, can hurt worst-seed stability). |
| **`rf_min_leaf`** | `min_samples_leaf`; larger values **smooth** the forest. |
| **Fixed in experiments** | **LR** `C=100`, **NB** `α=0.9`, NB/CNB **blend=1.0**, same `NB_FEAT_PARAMS` / TF-IDF settings as the main pipeline. |
| **`seeds: short` vs `long`** | **`short`** = 6 split seeds `[1,7,13,21,42,84]` (fast screens). **`long`** = 10 seeds (+ `97,123,256,512`) for stabler estimates. |
| **Metric** | **80/20 stacking**: train+val = 80% (OOF meta + refit), **test** = held-out 20% **person-level** split; accuracy per seed, then **min / mean / max** over seeds. |

`stacking_ensemble.py` still does **Stage‑1 CV** for LR/NB/RF grids on a single pipeline run; the experiment script uses **fixed** LR/NB/RF except the RF fields in each preset, so comparisons isolate **meta + RF** effects.

---

## Presets in code (default rows)

| Id | Intent |
|----|--------|
| **A** | Original-style **strong meta** (`C=1e4`) + **200 trees, deep RF**. |
| **B** | **More regularised RF** (400 / depth 12 / leaf 4) + **meta `C=1`**. |
| **C** | **Floor-focused**: **meta `C=0.5`**, RF **200 / unlimited depth / leaf 1** — candidate for best **min** on short seeds. |
| **D** | Same RF as **C**, **meta `C=1`**. |
| **E** | Same RF as **B**, **6-seed** screen only (quick comparison to older “fast eval”). |

---

## Recorded results (from earlier runs; re-run to reproduce exactly)

Numbers depend on **seed list** and code version; treat as **ballpark** and refresh with `python stacking_experiments.py`.

| Label | Setting (meta C, RF) | Seeds | min acc | mean acc | max acc | Notes |
|-------|----------------------|-------|---------|----------|---------|--------|
| Baseline | `1e4`, 200 / None / 1 | 10 | **0.914** | **0.934** | — | Old min-priority sweep vs “robust RF” candidates. |
| Robust RF + meta 1 | `1`, 400 / 12 / 4 | 10 | **0.917** | **0.930** | — | Better min, slightly lower mean. |
| **Chosen for stability** | **`0.5`, 200 / None / 1** | **6** | **0.923** | **0.938** | **0.960** | Quick multiseed grid (`RandomState` OOF); used as basis for **`final`** export hyperparameters. |
| Fast 6-seed screen | `1`, 400 / 12 / 4 | 6 | 0.920 | 0.936 | — | Aligns with preset **E**. |

**Takeaway:** tightening **meta** regularisation and adjusting **RF** traded a little **mean** for a higher **worst-seed** accuracy; the **`final`** branch freezes the chosen values and adds `pred.py` / `export_model.py` for MarkUs.

---

## Other files

| File | Role |
|------|------|
| `stacking_ensemble.py` | Full notebook-style run: Stage‑1 CV + OOF stacking + test report. |
| `pipeline.py`, `naive_bayes.py` | Features and cleaning. |
| `data_exploration.py` | EDA for reports. |

## Submission snapshot

```bash
git checkout final
```

---

*Stacking-ensemble heritage + parameter study on `cheng2`; frozen submit bundle on `final`.*
