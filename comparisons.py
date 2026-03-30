import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.utils.multiclass import unique_labels


BASELINE_ARRAYS = "preprocessed_arrays.npz"
INTER_ARRAYS = "preprocessed_arrays_interactions.npz"
STATE_PATH = "preprocess_state.pkl"


def load_state():
    with open(STATE_PATH, "rb") as f:
        return pickle.load(f)


def load_npz(path):
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"]


def metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_cm(ax, y_true, y_pred, title):
    labels = unique_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=[str(x) for x in labels]).plot(
        ax=ax, xticks_rotation=45, values_format="d"
    )
    ax.set_title(title)
    return cm


def compare_text_nb(state, base, inter):
    vocab = state["_vectorizer"].get_feature_names_out()
    n_text = len(vocab)
    n_other_base = base["X_train"].shape[1] - n_text
    n_other_inter = inter["X_train"].shape[1] - n_text

    # text slice always the last n_text columns in both variants
    Xtr_b = np.maximum(base["X_train"][:, n_other_base:], 0)
    Xva_b = np.maximum(base["X_val"][:, n_other_base:], 0)
    Xtr_i = np.maximum(inter["X_train"][:, n_other_inter:], 0)
    Xva_i = np.maximum(inter["X_val"][:, n_other_inter:], 0)

    nb_b = MultinomialNB().fit(Xtr_b, base["y_train"])
    nb_i = MultinomialNB().fit(Xtr_i, inter["y_train"])

    pred_b = nb_b.predict(Xva_b)
    pred_i = nb_i.predict(Xva_i)

    return {"pred_b": pred_b, "pred_i": pred_i, "m_b": metrics(base["y_val"], pred_b), "m_i": metrics(inter["y_val"], pred_i)}


def compare_all_gaussian_nb(base, inter):
    nb_b = GaussianNB().fit(base["X_train"], base["y_train"])
    nb_i = GaussianNB().fit(inter["X_train"], inter["y_train"])

    pred_b = nb_b.predict(base["X_val"])
    pred_i = nb_i.predict(inter["X_val"])

    return {"pred_b": pred_b, "pred_i": pred_i, "m_b": metrics(base["y_val"], pred_b), "m_i": metrics(inter["y_val"], pred_i)}


def compare_random_forest(base, inter, tree_counts=range(10, 201, 10)):
    tree_counts = list(tree_counts)
    acc_b, acc_i = [], []

    for n in tree_counts:
        rf_b = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1).fit(base["X_train"], base["y_train"])
        rf_i = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1).fit(inter["X_train"], inter["y_train"])
        acc_b.append(float(rf_b.score(base["X_val"], base["y_val"])))
        acc_i.append(float(rf_i.score(inter["X_val"], inter["y_val"])))

    best_n_b = tree_counts[int(np.argmax(acc_b))]
    best_n_i = tree_counts[int(np.argmax(acc_i))]

    rf_b = RandomForestClassifier(n_estimators=best_n_b, random_state=0, n_jobs=-1).fit(base["X_train"], base["y_train"])
    rf_i = RandomForestClassifier(n_estimators=best_n_i, random_state=0, n_jobs=-1).fit(inter["X_train"], inter["y_train"])

    pred_b = rf_b.predict(base["X_val"])
    pred_i = rf_i.predict(inter["X_val"])

    return {
        "tree_counts": tree_counts,
        "acc_curve_b": acc_b,
        "acc_curve_i": acc_i,
        "best_n_b": best_n_b,
        "best_n_i": best_n_i,
        "pred_b": pred_b,
        "pred_i": pred_i,
        "m_b": metrics(base["y_val"], pred_b),
        "m_i": metrics(inter["y_val"], pred_i),
    }


def main():
    state = load_state()
    idx_to_painting = {i: name for name, i in state["class_to_idx"].items()}

    Xtr_b, ytr_b, Xva_b, yva_b = load_npz(BASELINE_ARRAYS)
    Xtr_i, ytr_i, Xva_i, yva_i = load_npz(INTER_ARRAYS)

    base = {"X_train": Xtr_b, "y_train": ytr_b, "X_val": Xva_b, "y_val": yva_b}
    inter = {"X_train": Xtr_i, "y_train": ytr_i, "X_val": Xva_i, "y_val": yva_i}

    print("Validation-only comparisons (baseline vs +interactions)\n")
    print("Class mapping:", idx_to_painting)

    # ---- NB text-only ----
    text_res = compare_text_nb(state, base, inter)
    print("\n[MultinomialNB text-only]")
    print("baseline:", text_res["m_b"])
    print("+interactions:", text_res["m_i"])

    # ---- NB all features ----
    all_res = compare_all_gaussian_nb(base, inter)
    print("\n[GaussianNB all features]")
    print("baseline:", all_res["m_b"])
    print("+interactions:", all_res["m_i"])

    # ---- Random forest ----
    rf_res = compare_random_forest(base, inter)
    print("\n[RandomForest]")
    print(f"baseline best_n={rf_res['best_n_b']}, metrics={rf_res['m_b']}")
    print(f"+interactions best_n={rf_res['best_n_i']}, metrics={rf_res['m_i']}")

    # ---- Visualizations ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    plot_cm(axes[0, 0], yva_b, text_res["pred_b"], "Text NB baseline (val)")
    plot_cm(axes[0, 1], yva_i, text_res["pred_i"], "Text NB +interactions (val)")

    plot_cm(axes[1, 0], yva_b, all_res["pred_b"], "All-features GaussianNB baseline (val)")
    plot_cm(axes[1, 1], yva_i, all_res["pred_i"], "All-features GaussianNB +interactions (val)")

    plot_cm(
        axes[2, 0],
        yva_b,
        rf_res["pred_b"],
        f"RF baseline (val) n_estimators={rf_res['best_n_b']}",
    )
    plot_cm(
        axes[2, 1],
        yva_i,
        rf_res["pred_i"],
        f"RF +interactions (val) n_estimators={rf_res['best_n_i']}",
    )

    plt.tight_layout()
    plt.savefig("comparisons_confusion_matrices.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(rf_res["tree_counts"], rf_res["acc_curve_b"], marker="o", label="RF baseline")
    plt.plot(rf_res["tree_counts"], rf_res["acc_curve_i"], marker="o", label="RF +interactions")
    plt.xlabel("Number of Trees")
    plt.ylabel("Validation Accuracy")
    plt.title("Random Forest accuracy curves (validation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparisons_rf_accuracy_curves.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

