"""
Preprocessing and Transformations for CSC311 painting classification.
Supports both Method 1 (regular_split) and Method 2 (grouped_kfold_split).
Fit on train only; transform train/val/test with the same state.
State can be saved/loaded for pred.py and other models.
"""
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Column names (must match training_data.csv)
COL_ID = "unique_id"
COL_TARGET = "Painting"
COL_EMOTION = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
COL_DESC = "Describe how this painting makes you feel."
COL_SOMBRE = "This art piece makes me feel sombre."
COL_CONTENT = "This art piece makes me feel content."
COL_CALM = "This art piece makes me feel calm."
COL_UNEASY = "This art piece makes me feel uneasy."
COL_N_COLOURS = "How many prominent colours do you notice in this painting?"
COL_N_OBJECTS = "How many objects caught your eye in the painting?"
COL_PRICE = "How much (in Canadian dollars) would you be willing to pay for this painting?"
COL_ROOM = "If you could purchase this painting, which room would you put that painting in?"
COL_WHO = "If you could view this art in person, who would you want to view it with?"
COL_SEASON = "What season does this art piece remind you of?"
COL_FOOD = "If this painting was a food, what would be?"
COL_SOUNDTRACK = "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."

NUMERIC_COLS = [COL_EMOTION, COL_N_COLOURS, COL_N_OBJECTS]
LIKERT_COLS = [COL_SOMBRE, COL_CONTENT, COL_CALM, COL_UNEASY]
MULTI_COLS = [COL_ROOM, COL_WHO, COL_SEASON]
TEXT_COLS = [COL_DESC, COL_FOOD, COL_SOUNDTRACK]

REQUIRED_FOR_CLEAN = [COL_ID, COL_TARGET, COL_EMOTION, COL_N_COLOURS, COL_N_OBJECTS] + LIKERT_COLS

CLIP_PERCENTILE_DEFAULT = 97
TFIDF_MAX_FEATURES_DEFAULT = 6000
TFIDF_MIN_DF_DEFAULT = 2


def _extract_numeric(series, clip_max=None, impute=None):
    fill = float(impute) if impute is not None else 0.0
    out = []
    for v in series:
        if pd.isna(v):
            out.append(fill)
            continue
        m = re.search(r"\d+(?:,\d{3})*\.?\d*", str(v))
        if m:
            x = float(m.group().replace(",", ""))
            if clip_max is not None and x > clip_max:
                x = clip_max
            out.append(x)
        else:
            out.append(fill)
    return np.array(out, dtype=float)


def _extract_likert(series, impute=None):
    fill = int(round(np.clip(float(impute), 1, 5))) if impute is not None else 0
    out = []
    for v in series:
        if pd.isna(v):
            out.append(fill)
            continue
        m = re.search(r"^([1-5])", str(v).strip())
        out.append(int(m.group(1)) if m else fill)
    return np.array(out, dtype=float)


def _get_categories(series):
    cats = set()
    for v in series.dropna():
        for part in str(v).split(","):
            cats.add(part.strip())
    return sorted(cats)


def clean(df):
    """
    Basic cleaning before split. Drops rows missing required columns; fills text/multi with ''.
    Returns df_clean. Use this output for data_splitting.regular_split or grouped_kfold_split.
    """
    df_clean = df.dropna(subset=REQUIRED_FOR_CLEAN).copy()
    for c in TEXT_COLS:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna("")
    for c in MULTI_COLS:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna("")
    return df_clean


def fit_preprocess(
    train_df,
    clip_percentile=CLIP_PERCENTILE_DEFAULT,
    max_features=TFIDF_MAX_FEATURES_DEFAULT,
    min_df=TFIDF_MIN_DF_DEFAULT,
    impute="none",
):
    """
    Fit preprocessing on train_df only. Returns state dict for transform_df and save_state.
    impute: "none" (default) or "median" (reserved; median not implemented yet).
    """
    state = {
        "clip_percentile": clip_percentile,
        "max_features": max_features,
        "min_df": min_df,
        "impute": impute,
    }

    # Classes
    classes = sorted(train_df[COL_TARGET].unique())
    state["classes"] = classes
    state["class_to_idx"] = {c: i for i, c in enumerate(classes)}

    # Optional medians for impute="median" (reserved)
    num_medians = None
    likert_medians = None
    price_median = None
    if impute == "median":
        num_medians = []
        for c in NUMERIC_COLS:
            non_null = train_df[c].dropna()
            if len(non_null) == 0:
                num_medians.append(0.0)
            else:
                vals = _extract_numeric(non_null)
                num_medians.append(float(np.median(vals)))
        likert_medians = []
        for c in LIKERT_COLS:
            non_null = train_df[c].dropna()
            if len(non_null) == 0:
                likert_medians.append(3.0)
            else:
                vals = _extract_likert(non_null)
                likert_medians.append(float(np.median(vals)))
        price_non_null = train_df[COL_PRICE].dropna()
        price_median = float(np.median(_extract_numeric(price_non_null))) if len(price_non_null) > 0 else 0.0
    state["num_medians"] = num_medians
    state["likert_medians"] = likert_medians
    state["price_median"] = price_median

    # Numeric: clip + mean/std (on train)
    num_means, num_stds, num_clips = [], [], []
    for j, c in enumerate(NUMERIC_COLS):
        raw = _extract_numeric(train_df[c], impute=num_medians[j] if num_medians else None)
        clip_val = np.percentile(raw[raw > 0] if np.any(raw > 0) else raw, clip_percentile)
        if clip_val == 0:
            clip_val = 1.0
        num_clips.append(float(clip_val))
        x = _extract_numeric(train_df[c], clip_max=clip_val, impute=num_medians[j] if num_medians else None)
        num_means.append(float(np.mean(x)))
        num_stds.append(float(np.std(x)) or 1.0)
    state["num_means"] = np.array(num_means)
    state["num_stds"] = np.array(num_stds)
    state["num_clips"] = np.array(num_clips)

    # Price
    price_raw = _extract_numeric(train_df[COL_PRICE], impute=price_median)
    price_clip = np.percentile(price_raw[price_raw > 0] if np.any(price_raw > 0) else price_raw, clip_percentile)
    if price_clip == 0:
        price_clip = 1.0
    state["price_clip"] = float(price_clip)
    price_train = _extract_numeric(train_df[COL_PRICE], clip_max=price_clip, impute=price_median)
    state["price_mean"] = float(np.mean(price_train))
    state["price_std"] = float(np.std(price_train)) or 1.0

    # Multi-hot categories
    state["room_cats"] = _get_categories(train_df[COL_ROOM])
    state["who_cats"] = _get_categories(train_df[COL_WHO])
    state["season_cats"] = _get_categories(train_df[COL_SEASON])

    # TF-IDF
    train_text = (
        train_df[COL_DESC].astype(str) + " "
        + train_df[COL_FOOD].astype(str) + " "
        + train_df[COL_SOUNDTRACK].astype(str)
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    vectorizer.fit(train_text)
    state["vocab"] = list(vectorizer.get_feature_names_out())
    state["idf"] = vectorizer.idf_.astype(float)
    state["_vectorizer"] = vectorizer  # for transform; not saved to disk

    return state


def transform_df(df, state, return_y=True):
    """
    Transform df using fitted state. Returns (X, y) if return_y and 'Painting' in df else (X, None).
    """
    n_rows = len(df)
    num_means = state["num_means"]
    num_stds = state["num_stds"]
    num_clips = state["num_clips"]
    num_medians = state.get("num_medians")
    likert_medians = state.get("likert_medians")
    price_median = state.get("price_median")

    # Numeric
    x_num_list = []
    for j, c in enumerate(NUMERIC_COLS):
        raw = _extract_numeric(df[c], clip_max=float(num_clips[j]), impute=num_medians[j] if num_medians else None)
        x_num_list.append((raw - num_means[j]) / num_stds[j])
    X_num = np.column_stack(x_num_list)

    # Likert
    likert_list = [
        _extract_likert(df[c], impute=likert_medians[j] if likert_medians else None).astype(float)
        for j, c in enumerate(LIKERT_COLS)
    ]
    X_likert = np.column_stack(likert_list)

    # Price
    pr = _extract_numeric(df[COL_PRICE], clip_max=state["price_clip"], impute=price_median)
    X_price = ((pr - state["price_mean"]) / state["price_std"]).reshape(-1, 1)

    # Multi-hot
    X_room = _multi_hot(df[COL_ROOM], state["room_cats"])
    X_who = _multi_hot(df[COL_WHO], state["who_cats"])
    X_season = _multi_hot(df[COL_SEASON], state["season_cats"])

    # TF-IDF
    text = (
        df[COL_DESC].astype(str) + " "
        + df[COL_FOOD].astype(str) + " "
        + df[COL_SOUNDTRACK].astype(str)
    )
    X_tfidf = state["_vectorizer"].transform(text).toarray()

    X = np.hstack([X_num, X_likert, X_price, X_room, X_who, X_season, X_tfidf])

    y = None
    if return_y and COL_TARGET in df.columns:
        class_to_idx = state["class_to_idx"]
        y = np.array([class_to_idx[c] for c in df[COL_TARGET]])

    return X, y


def _multi_hot(series, categories):
    n = len(series)
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((n, k), dtype=float)
    for i, v in enumerate(series):
        if pd.isna(v):
            continue
        for part in str(v).split(","):
            key = part.strip()
            if key in cat_to_idx:
                mat[i, cat_to_idx[key]] = 1.0
    return mat


def save_state(state, path_prefix="preprocess_state"):
    """Save state to path_prefix.npz and path_prefix.json (for pred.py and other models)."""
    # Don't persist sklearn vectorizer
    state_copy = {k: v for k, v in state.items() if not k.startswith("_")}
    np.savez(
        path_prefix + ".npz",
        num_means=state["num_means"],
        num_stds=state["num_stds"],
        num_clips=state["num_clips"],
        idf=state["idf"],
        price_mean=np.array([state["price_mean"]]),
        price_std=np.array([state["price_std"]]),
        price_clip=np.array([state["price_clip"]]),
        num_medians=np.array(state["num_medians"]) if state.get("num_medians") is not None else np.array([]),
        likert_medians=np.array(state["likert_medians"]) if state.get("likert_medians") is not None else np.array([]),
        price_median=np.array([state["price_median"]]) if state.get("price_median") is not None else np.array([0.0]),
    )
    json_dict = {
        "classes": state["classes"],
        "vocab": state["vocab"],
        "room_cats": state["room_cats"],
        "who_cats": state["who_cats"],
        "season_cats": state["season_cats"],
        "clip_percentile": state["clip_percentile"],
        "max_features": state["max_features"],
        "min_df": state["min_df"],
        "impute": state["impute"],
    }
    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=0)


def load_state(path_prefix="preprocess_state", attach_vectorizer=True):
    """
    Load state from path_prefix.npz and path_prefix.json.
    If attach_vectorizer=True (default), rebuilds TfidfVectorizer from vocab so transform_df works.
    Set attach_vectorizer=False for pred.py-style use (no sklearn; use vocab+idf manually).
    """
    data = np.load(path_prefix + ".npz", allow_pickle=False)
    with open(path_prefix + ".json", "r", encoding="utf-8") as f:
        j = json.load(f)
    state = {
        "num_means": data["num_means"],
        "num_stds": data["num_stds"],
        "num_clips": data["num_clips"],
        "idf": data["idf"],
        "price_mean": float(data["price_mean"][0]),
        "price_std": float(data["price_std"][0]),
        "price_clip": float(data["price_clip"][0]),
        "classes": j["classes"],
        "class_to_idx": {c: i for i, c in enumerate(j["classes"])},
        "vocab": j["vocab"],
        "room_cats": j["room_cats"],
        "who_cats": j["who_cats"],
        "season_cats": j["season_cats"],
        "clip_percentile": j["clip_percentile"],
        "max_features": j["max_features"],
        "min_df": j["min_df"],
        "impute": j["impute"],
    }
    num_medians = data["num_medians"]
    state["num_medians"] = list(num_medians) if len(num_medians) > 0 else None
    likert_medians = data["likert_medians"]
    state["likert_medians"] = list(likert_medians) if len(likert_medians) > 0 else None
    state["price_median"] = float(data["price_median"][0]) if len(data["price_median"]) > 0 else None
    if attach_vectorizer:
        v = TfidfVectorizer(
            vocabulary=j["vocab"],
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        v.fit([""])  # dummy fit so transform exists
        v.idf_ = data["idf"]
        state["_vectorizer"] = v
    else:
        state["_vectorizer"] = None
    return state


# ---------------------------------------------------------------------------
# Example: Method 1 and Method 2 usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_splitting import regular_split, grouped_kfold_split

    df = pd.read_csv("training_data.csv")
    df_clean = clean(df)
    print(f"After clean: {len(df_clean)} rows")

    # Method 1: 60/20/20
    train_df, val_df, test_df = regular_split(df_clean)
    state = fit_preprocess(train_df, clip_percentile=97, max_features=6000, min_df=2, impute="none")
    X_train, y_train = transform_df(train_df, state)
    X_val, y_val = transform_df(val_df, state)
    X_test, y_test = transform_df(test_df, state)
    print(f"Method 1 - Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    save_state(state, "preprocess_state")
    print("Saved preprocess_state.npz and preprocess_state.json")

    # Method 2: 5-fold (one fold demo)
    for fold_idx, (tr, va, te) in enumerate(grouped_kfold_split(df_clean, k=5)):
        st = fit_preprocess(tr, max_features=6000, min_df=2)
        Xtr, ytr = transform_df(tr, st)
        Xva, yva = transform_df(va, st)
        print(f"Method 2 Fold {fold_idx + 1} - Train {Xtr.shape}, Val {Xva.shape}")
        if fold_idx >= 0:
            break
