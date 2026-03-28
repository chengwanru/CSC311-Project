# CSC311 — `cheng2` (stacking development)

Stacking: **LR + NB/CNB + RF** → **meta logistic regression** on 5-fold person-level OOF probs (`stacking_ensemble.py`). MarkUs-ready code is on **`final`** only (`git checkout final`).

---

## Multiseed experiments (80/20 test acc over split seeds)

Same setup for both: **6 seeds** `[1,7,13,21,42,84]`, person-level 80% train+val / 20% test, fixed **LR C=100**, **NB α=0.9**, **NB blend=1.0**, **RF = 200 trees, max_depth unlimited, min_samples_leaf=1**. Only **meta `C`** differs.

| | meta `C` | min | mean | max |
|---|----------|-----|------|-----|
| **1** | 0.5 | **0.9233** | 0.9380 | 0.9602 |
| **2** | 4.0 | 0.9172 | **0.9385** | 0.9602 |

**1** — best **worst-seed** (min). **2** — best **average** (mean); min is lower.

Re-run: `python stacking_experiments.py` (presets **`1`** and **`2`** in `stacking_experiments.py`).

---

## Parameters we actually changed

- **`meta_c`** — regularisation on the meta logistic layer; **smaller ⇒ stronger** regularisation.
- **RF** — here both runs used **200 trees, no depth cap, `min_samples_leaf=1`**.

---

## Other files

- `stacking_experiments.py` — runs presets above; edit `PRESETS` to try more settings.
- `stacking_ensemble.py` — full train + Stage‑1 CV + stacking evaluation.
- `pipeline.py`, `naive_bayes.py`, `data_exploration.py` — data / features / EDA.

---

*Submit bundle: **`final`** branch.*
