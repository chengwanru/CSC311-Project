# CSC311-Project

## Preprocessing

**Run once** (from project root):

```bash
python run_preprocess.py
```

This creates `preprocess_state.npz` and `preprocess_state.json`. In your code, use `load_state("preprocess_state")` and `transform_df(df, config)` to get feature matrices. To get train/val/test DataFrames, use `clean(df)` and `regular_split(df_clean)` from `preprocessing` and `data_splitting`.

---

### Hyperparameters

- `clip_percentile`: 97 (default) — clip numeric/price at this percentile.
- `max_features`, `min_df`: TF-IDF; tune as hyperparameters.

### Missing values

- `impute="none"` (default): drop rows missing required columns (see `REQUIRED_FOR_CLEAN`); use 0 for remaining missing in numeric/Likert when building features.
- `impute="median"`: reserved; fills numeric/Likert/price with train-set medians (implemented in `fit_preprocess`; use if you prefer median over drop).