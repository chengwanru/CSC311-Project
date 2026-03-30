"""Feature column names matching preprocessing.transform_df X order (no pandas import)."""


def feature_names_from_state(state, add_interactions=False):
    """
    Human-readable names for columns of X in the same order as transform_df
    (numeric → likert → [optional interactions] → multi-hot room/who/season → TF-IDF).
    """
    names = [
        "emotion_intensity",
        "n_colours",
        "n_objects",
        "likert_sombre",
        "likert_content",
        "likert_calm",
        "likert_uneasy",
    ]
    if add_interactions:
        names.extend(["content_x_calm", "uneasy_x_sombre"])
    for c in state["room_cats"]:
        names.append(f"room:{c}")
    for c in state["who_cats"]:
        names.append(f"who:{c}")
    for c in state["season_cats"]:
        names.append(f"season:{c}")
    for t in state["_vectorizer"].get_feature_names_out():
        names.append(f"tfidf:{t}")
    return names
