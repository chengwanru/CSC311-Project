# CSC311 — `cheng2` (stacking development)

This branch is **based on the original `stacking-ensemble` work**: the same **stacking** idea (**LR + NB/CNB + RF** → **meta logistic regression** on 5-fold person-level OOF probabilities), implemented in `stacking_ensemble.py` with shared `pipeline.py` / `naive_bayes.py`.

On top of that baseline, we **ran extra experiments**—mostly changing **meta `C`** and **RF** settings and measuring **min / mean / max** test accuracy over several split seeds (`stacking_experiments.py`). The table below is that follow-up study; the **MarkUs-ready** snapshot (fixed hyperparameters, `pred.py`, export) is on **`final`** (`git checkout final`).

---

## Multiseed experiments (80/20 test acc over split seeds)

Same setup for both: **6 seeds** `[1,7,13,21,42,84]`, person-level 80% train+val / 20% test, fixed **LR C=100**, **NB α=0.9**, **NB blend=1.0**, **RF = 200 trees, max_depth unlimited, min_samples_leaf=1**. Only **meta `C`** differs.

| | meta `C` | min | mean | max |
|---|----------|-----|------|-----|
| **1** | 0.5 | **0.9233** | 0.9380 | 0.9602 |
| **2** | 4.0 | 0.9172 | **0.9385** | 0.9602 |

**1** — best **min**. **2** — best **mean** (lower min).

Re-run: `python stacking_experiments.py` (presets `1` and `2` in `stacking_experiments.py`).

---

## Parameters we changed

- **meta `C`** — regularisation on the meta logistic layer; smaller ⇒ stronger regularisation.
- **RF** — both runs above used **200 trees, no depth cap, `min_samples_leaf=1`**.

---

## Other files

- `stacking_experiments.py` — runs presets above; edit `PRESETS` to try more settings.
- `stacking_ensemble.py` — full train + Stage‑1 CV + stacking evaluation.
- `pipeline.py`, `naive_bayes.py`, `data_exploration.py` — data / features / EDA.

---

*Submission bundle: **`final`** branch.*
