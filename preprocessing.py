"""
preprocessing.py
================
Cleans and preprocesses the full painting survey dataset into a single CSV.

What this script does:
  - Removes fully-blank students
  - Parses Likert strings ('4 - Agree') into floats 1-5 (NaN if missing)
  - Cleans and log-transforms the price column (NaN if unparseable)
  - Clips numeric columns to survey / fixed bounds only (no data-driven caps)
  - Cleans categorical columns (trims whitespace, keeps as strings)
  - Keeps text columns raw (nulls left as empty string)
  - Retains unique_id for traceability

What this script does NOT do:
  - Upper clipping of colour/object counts from percentiles -- fitted on train in build_features()
  - Imputation -- handled in build_features() fitted on train fold only
  - Train/val/test splitting -- handled in modelling scripts
  - One-hot / binary encoding -- handled in modelling scripts
  - Text vectorization -- handled in modelling scripts

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

# Hard clip bounds (lower, upper). Upper None = no upper clip here (train-fold 99th pct in naive_bayes).
NUMERIC_CLIP = {
    "emotion_intensity": (1.0, 10.0),
    "colours_noticed"  : (1.0, None),
    "objects_noticed"  : (1.0, None),
}

# ============================================================
# PARSE HELPERS
# ============================================================

def parse_likert(series):
    """'4 - Agree' -> 4.0  |  NaN / unparseable -> NaN"""
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
    Returns NaN where no number is found.
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
    Trims whitespace from each comma-separated token.
    'Bedroom, Living room' -> 'Bedroom,Living room'.
    Nulls stay NaN.
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
    Removes all rows belonging to a student whose responses are entirely
    blank (all feature columns NaN or empty string) across all paintings.
    Returns (cleaned_df, list_of_removed_ids).
    """
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]

    def row_is_blank(row):
        for col in feature_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str) and val.strip() == "":
                continue
            return False
        return True

    row_blank        = df.apply(row_is_blank, axis=1)
    blank_per_person = row_blank.groupby(df[ID_COL]).transform("all")
    blank_ids        = df.loc[blank_per_person, ID_COL].unique().tolist()
    cleaned          = df[~blank_per_person].copy().reset_index(drop=True)
    return cleaned, blank_ids


# ============================================================
# TRANSFORM  (parse + clip only, no imputation)
# ============================================================

def transform(df):
    """
    Parses and clips all columns. Nulls are intentionally left in
    Likert, numeric, and price columns -- imputation is deferred to
    build_features() in the modelling script, where it is fitted
    on the training fold only to avoid data leakage.
    """
    df  = df.copy()
    out = pd.DataFrame(index=df.index)

    out[ID_COL]     = df[ID_COL]
    out[TARGET_COL] = df[TARGET_COL]

    # ── Likert: parse string -> float 1-5, NaN stays NaN ─────
    for col, short in LIKERT_COLS.items():
        out[short] = parse_likert(df[col]).clip(lower=1, upper=5)

    # ── Numeric: coerce -> clip, NaN stays NaN ───────────────
    for col, short in NUMERIC_COLS.items():
        numeric  = pd.to_numeric(df[col], errors="coerce")
        lo, hi   = NUMERIC_CLIP[short]
        if hi is None:
            out[short] = numeric.clip(lower=lo)
        else:
            out[short] = numeric.clip(lower=lo, upper=hi)

    # ── Price: parse + log1p, NaN stays NaN ──────────────────
    out["price_log1p"] = parse_price(df[PRICE_COL]).round(4)

    # ── Categorical: trim whitespace, NaN stays NaN ──────────
    for col, short in CATEGORY_COLS.items():
        out[short] = clean_category(df[col])

    # ── Text: raw string, null -> empty string ────────────────
    for col, short in TEXT_COLS.items():
        out[short] = df[col].fillna("").str.strip()

    return out


# ============================================================
# SUMMARY REPORT
# ============================================================

def print_summary(df_raw, df_clean, n_removed_ids, n_removed_rows):
    print("\n" + "=" * 65)
    print("PREPROCESSING SUMMARY")
    print("=" * 65)

    print(f"\nROW COUNT")
    print(f"  Raw             : {len(df_raw)}  ({df_raw[ID_COL].nunique()} persons)")
    print(f"  Blank removed   : {n_removed_ids} persons  ({n_removed_rows} rows)")
    print(f"  Final           : {len(df_clean)}  ({df_clean[ID_COL].nunique()} persons)")

    print(f"\nNULL COUNTS IN OUTPUT  (will be imputed in build_features)")
    for short in list(LIKERT_COLS.values()) + list(NUMERIC_COLS.values()) + ["price_log1p"]:
        n = df_clean[short].isna().sum()
        if n > 0:
            print(f"  {short:<20}  {n:>3} nulls")
    for short in CATEGORY_COLS.values():
        n = df_clean[short].isna().sum()
        if n > 0:
            print(f"  {short:<20}  {n:>3} nulls  (kept -- no imputation for strings)")

    print(f"\nOUTLIER CLIPPING")
    print(f"  emotion_intensity  clipped to [1, 10]  (survey bounds)")
    print(f"  colours_noticed, objects_noticed  lower bound 1 only; upper 99th pct in modelling (train fit)")
    print(f"  price              hard cap at ${PRICE_CAP:,} before log transform")

    print(f"\nOUTPUT COLUMNS  ({len(df_clean.columns)} total)")
    for c in df_clean.columns:
        print(f"  {c:<30}  {str(df_clean[c].dtype)}")
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

    df_filtered, blank_ids = drop_blank_students(df)
    n_removed_ids  = len(blank_ids)
    n_removed_rows = len(df) - len(df_filtered)
    print(f"  Removed {n_removed_ids} fully-blank persons ({n_removed_rows} rows)\n")

    print("Transforming...")
    df_clean = transform(df_filtered)

    print_summary(df, df_clean, n_removed_ids, n_removed_rows)

    df_clean.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}  ({len(df_clean)} rows x {len(df_clean.columns)} columns)")
