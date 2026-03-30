"""
Preprocess and save train/val/test arrays for Random Forest

- Loads CSV
- Cleans
- Removes blank users
- Splits
- Fits preprocessing
- Transforms train/val/test
- Saves state + arrays
"""

import pandas as pd
import pickle
import numpy as np
from data_splitting import regular_split
from preprocessing import COL_SEASON, COL_TARGET, clean, fit_preprocess, transform_df

CSV_PATH = "training_data.csv"
STATE_PATH = "preprocess_state.pkl"
ARRAY_PATH = "preprocessed_arrays.npz"
ARRAY_PATH_INTERACTIONS = "preprocessed_arrays_interactions.npz"

def remove_blank_users(df):
    COL_ID = "unique_id"
    COL_TARGET = "Painting"

    feature_cols = [c for c in df.columns if c not in [COL_ID, COL_TARGET]]

    def row_is_blank(row):
        for col in feature_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str) and val.strip() == "":
                continue
            return False
        return True

    row_blank = df.apply(row_is_blank, axis=1)
    blank_per_person = row_blank.groupby(df[COL_ID]).transform("all")

    return df[~blank_per_person].copy().reset_index(drop=True)


def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded")

    # CLEAN
    print("Cleaning...")
    df_clean = clean(df)

    # REMOVE FULLY BLANK USERS
    print("Removing fully blank users...")
    before = len(df_clean)
    df_clean = remove_blank_users(df_clean)
    after = len(df_clean)
    print(f"  Removed {before - after} rows")

    # SEASON x PAINTING FREQUENCIES (from cleaned data)
    # Note: season responses can be comma-separated; count each token.
    season_df = df_clean[[COL_TARGET, COL_SEASON]].copy()
    season_df[COL_SEASON] = (
        season_df[COL_SEASON]
        .fillna("")
        .astype(str)
        .str.split(",")
    )
    season_df = season_df.explode(COL_SEASON, ignore_index=True)
    season_df[COL_SEASON] = season_df[COL_SEASON].astype(str).str.strip()
    season_df = season_df[season_df[COL_SEASON] != ""]

    def normalize_season(s: str) -> str:
        sl = s.strip().lower()
        if sl.startswith("spr"):
            return "Spring"
        if sl.startswith("sum"):
            return "Summer"
        if sl.startswith("win"):
            return "Winter"
        if sl.startswith("fal") or sl.startswith("aut"):
            return "Fall"
        return s.strip().title()

    season_df["Season"] = season_df[COL_SEASON].map(normalize_season)
    freq = pd.crosstab(season_df["Season"], season_df[COL_TARGET])
    freq = freq.reindex(["Spring", "Summer", "Fall", "Winter"]).fillna(0).astype(int)
    print("\nSeason × Painting frequency table (cleaned data):")
    print(freq.to_string())

    # SPLIT
    print("Splitting (60/20/20)...")
    train_df, val_df, test_df = regular_split(df_clean)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # FIT PREPROCESS
    print("Fitting preprocessing...")
    state = fit_preprocess(train_df, max_features=6000, min_df=2)

    # TRANSFORM
    print("Transforming train/val/test sets (baseline)...")
    X_train, y_train = transform_df(train_df, state, add_interactions=False)
    X_val, y_val = transform_df(val_df, state, add_interactions=False)
    X_test, y_test = transform_df(test_df, state, add_interactions=False)

    print("Transforming train/val/test sets (+ interactions)...")
    X_train_i, y_train_i = transform_df(train_df, state, add_interactions=True)
    X_val_i, y_val_i = transform_df(val_df, state, add_interactions=True)
    X_test_i, y_test_i = transform_df(test_df, state, add_interactions=True)

    # SAVE STATE
    print("Saving state...")
    with open(STATE_PATH, "wb") as f:
        pickle.dump(state, f)

    # SAVE ARRAYS (baseline)
    print("Saving preprocessed arrays (baseline)...")
    np.savez(ARRAY_PATH,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    # SAVE ARRAYS (+ interactions)
    print("Saving preprocessed arrays (+ interactions)...")
    np.savez(ARRAY_PATH_INTERACTIONS,
             X_train=X_train_i, y_train=y_train_i,
             X_val=X_val_i, y_val=y_val_i,
             X_test=X_test_i, y_test=y_test_i)

    print(f"\nDone! Saved {STATE_PATH}, {ARRAY_PATH}, and {ARRAY_PATH_INTERACTIONS}")


if __name__ == "__main__":
    main()