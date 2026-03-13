"""
One-shot script: load data → clean → split (Method 1) → fit preprocess → save config.
Run once to generate preprocess_state.npz and preprocess_state.json. Teammates can
then load_state() and transform_df() in their code without running steps manually.
"""
import pandas as pd
from data_splitting import regular_split
from preprocessing import clean, fit_preprocess, save_state

CSV_PATH = "training_data.csv"
OUTPUT_PREFIX = "preprocess_state"

def main():
    df = pd.read_csv(CSV_PATH)
    df_clean = clean(df)
    train_df, val_df, test_df = regular_split(df_clean)
    config = fit_preprocess(train_df, clip_percentile=97, max_features=6000, min_df=2, impute="none")
    save_state(config, OUTPUT_PREFIX)
    print(f"Done. Saved {OUTPUT_PREFIX}.npz and {OUTPUT_PREFIX}.json")

if __name__ == "__main__":
    main()
