# Appendix code (models B & C + multiseed table helper)

These scripts are **not** used for MarkUs prediction. The **submitted model is A** in the repository root: `stacking_ensemble.py` (evaluation), `export_model.py` + `pred.py` (submission).

Include this folder in **code.zip** as evidence of alternative approaches explored for the report appendix.

| File | Role |
|------|------|
| `stacking_multiseed.py` | **Model B** — multiple random seeds for each base model; meta-logistic on **45-dim** OOF probabilities; Stage-1 CV for base hyperparameters. |
| `stacking_ensemble_multiRF.py` | **Model C** — average **RF** predictions over several `random_state` values; **9-dim** meta features; Stage-1 CV. |
| `stacking_experiments.py` | Sweeps **meta C** (and presets) with **fixed** base hyperparameters; produces min/mean/max over person-level split seeds (aligns with report tables for model A sensitivity). |

Run from the **repository root** (so `pipeline.py` resolves):

```bash
python appendix_code/stacking_experiments.py
python appendix_code/stacking_multiseed.py
python appendix_code/stacking_ensemble_multiRF.py
```

Each file prepends the repo root to `sys.path` so imports work from this subdirectory.
