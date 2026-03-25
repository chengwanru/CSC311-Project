"""Feature column names matching preprocessing.transform_df X order (no pandas import)."""


def feature_names_from_state(state):
    """
    Human-readable names for columns of X in the same order as transform_df
    (numeric → likert → price → multi-hot room/who/season → TF-IDF).
    """
    names = [
        "emotion_intensity",
        "n_colours",
        "n_objects",
        "likert_sombre",
        "likert_content",
        "likert_calm",
        "likert_uneasy",
        "log_price",
    ]
    for c in state["room_cats"]:
        names.append(f"room:{c}")
    for c in state["who_cats"]:
        names.append(f"who:{c}")
    for c in state["season_cats"]:
        names.append(f"season:{c}")
    for t in state["_vectorizer"].get_feature_names_out():
        names.append(f"tfidf:{t}")
    return names
