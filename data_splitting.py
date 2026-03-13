import numpy as np
import pandas as pd

def regular_split(df, id_col="unique_id", random_state=42):
    """
    Method 1: 60% / 20% / 20% split by person (unique_id).
    Ensures the test set size matches the K-Fold methodology for fair comparison.
    Returns (train_df, val_df, test_df).
    """
    # Explicitly remove rows without an ID first to prevent silent data loss
    df_clean = df.dropna(subset=[id_col]).copy()
    
    rng = np.random.RandomState(random_state)

    # Get and shuffle unique ids
    unique_ids = df_clean[id_col].unique()
    rng.shuffle(unique_ids)

    n_ids = len(unique_ids)

    # Pythonic integer casting & matching the 20% test holdout
    n_train = int(n_ids * 0.60)
    n_val   = int(n_ids * 0.20)
    n_test  = n_ids - n_train - n_val # Guarantees all IDs are assigned

    train_ids = unique_ids[:n_train]
    val_ids   = unique_ids[n_train:n_train + n_val]
    test_ids  = unique_ids[n_train + n_val:]

    # Create the final splits
    train_df = df_clean[df_clean[id_col].isin(train_ids)].copy()
    val_df   = df_clean[df_clean[id_col].isin(val_ids)].copy()
    test_df  = df_clean[df_clean[id_col].isin(test_ids)].copy()

    return train_df, val_df, test_df


def grouped_kfold_split(df, id_col="unique_id", k=5, random_state=42):
    """
    Method 2: 5-fold grouped CV with a held-out test set.
    - Randomly permute the unique ids.
    - First 80% ids -> train/validation pool (for 5-fold CV).
    - Remaining 20% ids -> fixed test set.
    Yields 5 tuples: (train_df, val_df, test_df) for each fold.
    """
    rng = np.random.RandomState(random_state)

    # Get and shuffle unique ids
    unique_ids = df[id_col].dropna().unique()
    rng.shuffle(unique_ids)

    n_ids = len(unique_ids)

    # Split 80/20
    n_trainval_ids = int(n_ids * 0.80)
    
    trainval_ids = unique_ids[:n_trainval_ids]
    test_ids = unique_ids[n_trainval_ids:]

    # Fixed test set (same for all folds)
    test_df = df[df[id_col].isin(test_ids)].copy()

    # THE FIX: Use array_split to handle uneven division perfectly
    folds = np.array_split(trainval_ids, k)

    for i in range(k):
        val_ids = folds[i]
        # Concatenate all folds EXCEPT the current validation fold
        train_ids = np.concatenate([folds[j] for j in range(k) if j != i])

        train_df = df[df[id_col].isin(train_ids)].copy()
        val_df   = df[df[id_col].isin(val_ids)].copy()

        yield train_df, val_df, test_df


# ==========================================
# Demonstration and Printout Code
# ==========================================

if __name__ == "__main__":
    # Load your specific dataset
    file_path = "training_data.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"--- Original Dataset Loaded ---")
        print(f"Total Rows: {len(df)}")
        print(f"Total Unique Students: {df['unique_id'].nunique()}\n")
        
        # ------------------------------------------
        # Test Method 1: Regular Split
        # ------------------------------------------
        print("==========================================")
        print("METHOD 1: STATIC GROUP SPLIT (60/20/20)")
        print("==========================================")
        train_df, val_df, test_df = regular_split(df)
        
        print(f"TRAIN SET: {train_df['unique_id'].nunique()} students | {len(train_df)} rows")
        print(f"VAL SET  : {val_df['unique_id'].nunique()} students | {len(val_df)} rows")
        print(f"TEST SET : {test_df['unique_id'].nunique()} students | {len(test_df)} rows")
        print(f"Total Data Accounted For: {len(train_df) + len(val_df) + len(test_df)} rows\n")

        # ------------------------------------------
        # Test Method 2: Grouped K-Fold Split
        # ------------------------------------------
        print("==========================================")
        print("METHOD 2: GROUP K-FOLD (k=5) + TEST HOLDOUT")
        print("==========================================")
        
        fold_number = 1
        for train_fold, val_fold, test_holdout in grouped_kfold_split(df, k=5):
            print(f"--- FOLD {fold_number} ---")
            print(f"  Training   (Folds 1-4) : {train_fold['unique_id'].nunique()} students | {len(train_fold)} rows")
            print(f"  Validation (Current)   : {val_fold['unique_id'].nunique()} students | {len(val_fold)} rows")
            print(f"  Test Set   (Held-out)  : {test_holdout['unique_id'].nunique()} students | {len(test_holdout)} rows")
            
            # Sanity check to ensure no data leakage between Train and Val
            train_ids_set = set(train_fold['unique_id'].unique())
            val_ids_set = set(val_fold['unique_id'].unique())
            overlap = train_ids_set.intersection(val_ids_set)
            
            if len(overlap) > 0:
                print(f"  [WARNING] Data Leakage Detected! {len(overlap)} IDs overlap.")
            else:
                print("  [SUCCESS] No data leakage between Train and Validation.")
            
            print("-" * 40)
            fold_number += 1

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please ensure the file is in the same directory as this script.")