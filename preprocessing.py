"""
preprocessing.py
================
Cleans and preprocesses the full painting survey dataset into a single CSV.

What this script does:
  - Parses Likert strings ('4 - Agree') into integers 1-5
  - Cleans and log-transforms the price column
  - Imputes missing values using per-painting medians
  - Clips numeric outliers (survey bounds + 99th percentile)
  - Cleans multi-label categorical columns (trims whitespace, keeps as strings)
  - Keeps text columns raw (nulls to empty string)
  - Retains unique_id for traceability

What this script does NOT do:
  - Train/val/test splitting  -- handled in modelling scripts
  - One-hot / binary encoding -- handled in modelling scripts
  - Text vectorization        -- handled in modelling scripts

Output:
  preprocessed.csv

Usage:
  python preprocessing.py
"""

import re
import numpy as np
import pandas as pd

# ============================================================
# COLUMN DEFINITIONS
# ============================================================

TARGET_COL = "Painting"
ID_COL     = "unique_id"

LIKERT_COLS = {
    "This art piece makes me feel sombre."  : "sombre",
    "This art piece makes me feel content." : "content",
    "This art piece makes me feel calm."    : "calm",
    "This art piece makes me feel uneasy."  : "uneasy",
}

NUMERIC_COLS = {
    "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?": "emotion_intensity",
    "How many prominent colours do you notice in this painting?"                  : "colours_noticed",
    "How many objects caught your eye in the painting?"                           : "objects_noticed",
}

PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"
PRICE_CAP = 10_000

CATEGORY_COLS = {
    "If you could purchase this painting, which room would you put that painting in?" : "room",
    "If you could view this art in person, who would you want to view it with?"       : "companion",
    "What season does this art piece remind you of?"                                  : "season",
}

TEXT_COLS = {
    "Describe how this painting makes you feel."                                                                   : "text_description",
    "If this painting was a food, what would be?"                                                                  : "text_food",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting." : "text_soundtrack",
}

# Numeric clipping bounds
NUMERIC_CLIP = {
    "emotion_intensity": (1.0, 10.0),  # hard survey bounds
    "colours_noticed"  : (1.0, None),  # upper cap = 99th percentile
    "objects_noticed"  : (1.0, None),  # upper cap = 99th percentile
}
NUMERIC_99PCT_COLS = ["colours_noticed", "objects_noticed"]

# ============================================================
# PARSE HELPERS
# ============================================================

def parse_likert(series):
    """'4 - Agree' -> 4.0   |   NaN / unparseable -> NaN"""
    def _p(x):
        if pd.isna(x):
            return np.nan
        m = re.match(r"^(\d)", str(x).strip())
        return float(m.group(1)) if m else np.nan
    return series.map(_p)


def parse_price(series, cap=PRICE_CAP):
    """
    Extracts the first number from a messy free-text price field,
    caps at PRICE_CAP, then applies log1p transform.
    Returns a float Series (NaN where no number found).
    """
    def _p(x):
        if pd.isna(x):
            return np.nan
        s = str(x).lower().replace(",", "").replace(" ", "")
        nums = re.findall(r"\d+\.?\d*", s)
        if not nums:
            return np.nan
        return min(float(nums[0]), cap)
    return np.log1p(series.map(_p))


def clean_category(series):
    """
    Trims whitespace from each comma-separated token so values like
    'Bedroom, Living room' become 'Bedroom,Living room'.
    Nulls are left as NaN.
    """
    def _clean(x):
        if pd.isna(x):
            return np.nan
        tokens = [t.strip() for t in str(x).split(",") if t.strip()]
        return ",".join(tokens)
    return series.map(_clean)


# ============================================================
# DROP FULLY BLANK STUDENTS
# ============================================================

def drop_blank_students(df):
    """
    Removes all rows belonging to a student (unique_id) whose responses
    are entirely blank across every painting they were shown.

    A student is considered fully blank if every response column
    (i.e. all columns except unique_id and Painting) is NaN across
    all of their rows. Such students contribute zero information.

    Students with at least one non-null response in any row are kept.

    Returns (cleaned_df, list_of_removed_ids).
    """
    response_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]

    blank_ids = [
        uid for uid, grp in df.groupby(ID_COL)
        if grp[response_cols].isnull().all().all()
    ]

    cleaned = df[~df[ID_COL].isin(blank_ids)].copy()
    return cleaned, blank_ids


# ============================================================
# FIT IMPUTATION PARAMETERS
# ============================================================

def fit_params(df):
    """
    Computes per-painting medians for imputation and 99th-percentile
    caps for outlier clipping from the full dataset.
    """
    params    = {}
    paintings = df[TARGET_COL].unique()

    # Likert medians per painting
    params["likert_medians"] = {}
    for col, short in LIKERT_COLS.items():
        parsed = parse_likert(df[col])
        per_painting = {}
        for p in paintings:
            mask   = df[TARGET_COL] == p
            median = parsed[mask].median()
            per_painting[p] = median if not np.isnan(median) else parsed.median()
        params["likert_medians"][short] = per_painting

    # Numeric medians per painting
    params["numeric_medians"] = {}
    for col, short in NUMERIC_COLS.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        per_painting = {}
        for p in paintings:
            mask   = df[TARGET_COL] == p
            median = numeric[mask].median()
            per_painting[p] = median if not np.isnan(median) else numeric.median()
        params["numeric_medians"][short] = per_painting

    # 99th-percentile caps
    params["numeric_99pct"] = {}
    for col, short in NUMERIC_COLS.items():
        if short in NUMERIC_99PCT_COLS:
            numeric = pd.to_numeric(df[col], errors="coerce")
            # Apply ceiling so the cap is a whole number — these are count
            # columns (objects/colours noticed) and can only be integers.
            # Ceiling rather than rounding avoids clipping any legitimate value.
            params["numeric_99pct"][short] = float(np.ceil(numeric.quantile(0.99)))

    # Price medians per painting (in log1p space)
    log_prices = parse_price(df[PRICE_COL])
    params["price_medians"] = {}
    for p in paintings:
        mask   = df[TARGET_COL] == p
        median = log_prices[mask].median()
        params["price_medians"][p] = median if not np.isnan(median) else log_prices.median()

    return params


# ============================================================
# TRANSFORM
# ============================================================

def transform(df, params):
    """
    Applies all cleaning and imputation steps.
    Returns a human-readable dataframe -- no encoding performed.
    """
    df  = df.copy()
    out = pd.DataFrame(index=df.index)

    # Keep ID and target
    out[ID_COL]     = df[ID_COL]
    out[TARGET_COL] = df[TARGET_COL]

    # Likert columns -> integer 1-5
    for col, short in LIKERT_COLS.items():
        parsed  = parse_likert(df[col])
        imputed = parsed.copy()
        for p, med in params["likert_medians"][short].items():
            mask = (df[TARGET_COL] == p) & imputed.isna()
            imputed[mask] = med
        out[short] = imputed.clip(lower=1, upper=5).astype(int)

    # Numeric columns -> clipped floats
    for col, short in NUMERIC_COLS.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        imputed = numeric.copy()
        for p, med in params["numeric_medians"][short].items():
            mask = (df[TARGET_COL] == p) & imputed.isna()
            imputed[mask] = med
        lo, hi = NUMERIC_CLIP.get(short, (None, None))
        if short in NUMERIC_99PCT_COLS:
            hi = params["numeric_99pct"][short]
        out[short] = imputed.clip(lower=lo, upper=hi).round(1)

    # Price -> log1p float
    log_prices    = parse_price(df[PRICE_COL])
    imputed_price = log_prices.copy()
    for p, med in params["price_medians"].items():
        mask = (df[TARGET_COL] == p) & imputed_price.isna()
        imputed_price[mask] = med
    out["price_log1p"] = imputed_price.round(4)

    # Categorical columns -> cleaned strings, kept as-is
    for col, short in CATEGORY_COLS.items():
        out[short] = clean_category(df[col])

    # Text columns -> raw string, null -> empty string
    for col, short in TEXT_COLS.items():
        out[short] = df[col].fillna("").str.strip()

    return out


# ============================================================
# SUMMARY REPORT
# ============================================================

def print_summary(df_raw, df_filtered, df_clean, params, n_removed_ids, n_removed_rows):
    print("\n" + "=" * 65)
    print("PREPROCESSING SUMMARY")
    print("=" * 65)

    print(f"\nROW COUNT")
    print(f"  Raw              : {len(df_raw)}  ({df_raw[ID_COL].nunique()} persons)")
    print(f"  Blank persons    : {n_removed_ids} removed  ({n_removed_rows} rows dropped)")
    print(f"  After filtering  : {len(df_filtered)}  ({df_filtered[ID_COL].nunique()} persons)")
    print(f"  Final cleaned    : {len(df_clean)}  ({df_clean[ID_COL].nunique()} persons)")

    print(f"\nNULL VALUES IMPUTED")
    for col, short in LIKERT_COLS.items():
        n = parse_likert(df_raw[col]).isna().sum()
        print(f"  {short:<20}  {n:>3} nulls  ->  per-painting median")
    for col, short in NUMERIC_COLS.items():
        n = pd.to_numeric(df_raw[col], errors="coerce").isna().sum()
        print(f"  {short:<20}  {n:>3} nulls  ->  per-painting median")
    n = parse_price(df_raw[PRICE_COL]).isna().sum()
    print(f"  {'price_log1p':<20}  {n:>3} nulls  ->  per-painting median (log scale)")
    for col, short in CATEGORY_COLS.items():
        n = df_raw[col].isna().sum()
        print(f"  {short:<20}  {n:>3} nulls  ->  kept as NaN (no imputation for strings)")

    print(f"\nOUTLIER CLIPPING")
    print(f"  emotion_intensity     clipped to [1, 10]  (survey-defined bounds)")
    for short in NUMERIC_99PCT_COLS:
        cap     = params["numeric_99pct"][short]
        raw_col = next(c for c, s in NUMERIC_COLS.items() if s == short)
        n       = (pd.to_numeric(df_raw[raw_col], errors="coerce") > cap).sum()
        print(f"  {short:<20}  99th pct cap = {cap:.1f}   ({n} values clipped)")
    print(f"  price                 hard cap at ${PRICE_CAP:,} before log transform")

    print(f"\nPER-PAINTING IMPUTATION VALUES -- LIKERT")
    for short in LIKERT_COLS.values():
        vals = params["likert_medians"][short]
        row  = f"  {short:<12}  " + \
               "  |  ".join(f"{p.split()[1]}: {v:.0f}" for p, v in vals.items())
        print(row)

    print(f"\nPER-PAINTING IMPUTATION VALUES -- NUMERIC")
    for short in NUMERIC_COLS.values():
        vals = params["numeric_medians"][short]
        row  = f"  {short:<20}  " + \
               "  |  ".join(f"{p.split()[1]}: {v:.1f}" for p, v in vals.items())
        print(row)

    print(f"\nCATEGORY COLUMNS  (kept as strings, whitespace trimmed)")
    for col, short in CATEGORY_COLS.items():
        sample = df_clean[short].dropna().iloc[0]
        print(f"  {short:<12}  sample: '{sample}'")

    print(f"\nOUTPUT COLUMNS  ({len(df_clean.columns)} total)")
    for c in df_clean.columns:
        dtype = str(df_clean[c].dtype)
        print(f"  {c:<30}  {dtype}")

    print(f"\nNULL CHECK")
    null_counts      = df_clean.isnull().sum()
    cols_with_nulls  = null_counts[null_counts > 0]
    numeric_nulls    = df_clean[list(LIKERT_COLS.values()) +
                                list(NUMERIC_COLS.values()) +
                                ["price_log1p"]].isnull().sum().sum()
    if numeric_nulls == 0:
        print("  All numeric/Likert/price columns: no nulls")
    if not cols_with_nulls.empty:
        for col, n in cols_with_nulls.items():
            print(f"  {col}: {n} nulls  (categorical -- intentionally kept)")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    DATA_PATH = "training_data.csv"
    OUT_PATH  = "preprocessed.csv"

    print(f"Loading '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows  |  {df[ID_COL].nunique()} students  |  "
          f"{df[TARGET_COL].nunique()} classes\n")

    # ── Remove persons with no valid entries across any of their rows ──
    # Must check raw data BEFORE imputation — imputation would fill blanks
    # with medians and make truly empty rows appear populated.
    feature_cols  = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]

    def row_is_blank(row):
        for col in feature_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str) and val.strip() == "":
                continue
            return False  # found at least one real value
        return True

    row_blank        = df.apply(row_is_blank, axis=1)
    blank_per_person = row_blank.groupby(df[ID_COL]).transform("all")
    df_filtered      = df[~blank_per_person].copy().reset_index(drop=True)

    n_removed_ids  = df[ID_COL].nunique() - df_filtered[ID_COL].nunique()
    n_removed_rows = len(df) - len(df_filtered)
    print(f"  Removed {n_removed_ids} fully-blank persons ({n_removed_rows} rows)\n")

    print("Fitting imputation parameters...")
    params = fit_params(df_filtered)

    print("Transforming...")
    df_clean = transform(df_filtered, params)

    print_summary(df, df_filtered, df_clean, params, n_removed_ids, n_removed_rows)

    df_clean.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}  ({len(df_clean)} rows x {len(df_clean.columns)} columns)")
