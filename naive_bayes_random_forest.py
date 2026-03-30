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
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels


STATE_PATH = "preprocess_state.pkl"
BASELINE_ARRAYS = "preprocessed_arrays.npz"
INTER_ARRAYS = "preprocessed_arrays_interactions.npz"


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


def main():
    # Compare baseline vs +interactions for:
    # - Hybrid: NB(text) + RF(structured)
    # - RF-only: RF(structured)
    # Validation only (no test set).
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)

    Xtr_b, ytr_b, Xva_b, yva_b = load_npz(BASELINE_ARRAYS)
    Xtr_i, ytr_i, Xva_i, yva_i = load_npz(INTER_ARRAYS)

    vocab = state["_vectorizer"].get_feature_names_out()
    n_text = len(vocab)
    n_other_b = Xtr_b.shape[1] - n_text
    n_other_i = Xtr_i.shape[1] - n_text

    # Text slices (same TF-IDF features, last n_text cols)
    Xtr_text_b = np.maximum(Xtr_b[:, n_other_b:], 0)
    Xva_text_b = np.maximum(Xva_b[:, n_other_b:], 0)
    Xtr_text_i = np.maximum(Xtr_i[:, n_other_i:], 0)
    Xva_text_i = np.maximum(Xva_i[:, n_other_i:], 0)

    # Structured slices (everything before TF-IDF; includes interactions in the "+interactions" arrays)
    Xtr_struct_b = Xtr_b[:, :n_other_b]
    Xva_struct_b = Xva_b[:, :n_other_b]
    Xtr_struct_i = Xtr_i[:, :n_other_i]
    Xva_struct_i = Xva_i[:, :n_other_i]

    # Hyperparameter grids (small on purpose; validation-only sweep)
    nb_smoothing = [0.1, 0.3, 1.0]
    n_estimators_list = [50, 100, 150, 200, 250]
    rf_max_features = ["sqrt", 0.5]
    rf_min_samples_leaf = [1, 2, 5]
    alphas = [round(a, 2) for a in np.linspace(0.0, 1.0, 21)]  # weight on NB

    hybrid_acc_b, rf_acc_b = [], []
    hybrid_acc_i, rf_acc_i = [], []

    hybrid_pred_by_n_i = {}
    rf_pred_by_n_i = {}
    hybrid_pred_by_n_b = {}
    rf_pred_by_n_b = {}

    # Track best weighted ensemble by macro-F1 (validation)
    best_b = {"macro_f1": -1.0}
    best_i = {"macro_f1": -1.0}

    # For the curves, keep a single representative RF setting (fast + readable curves)
    curve_rf_kwargs = {"max_features": "sqrt", "min_samples_leaf": 1}

    for n in n_estimators_list:
        # Curves use representative params
        rf_b_curve = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1, **curve_rf_kwargs).fit(
            Xtr_struct_b, ytr_b
        )
        rf_i_curve = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1, **curve_rf_kwargs).fit(
            Xtr_struct_i, ytr_i
        )
        rf_probs_b_curve = rf_b_curve.predict_proba(Xva_struct_b)
        rf_probs_i_curve = rf_i_curve.predict_proba(Xva_struct_i)
        pred_rf_b_curve = rf_b_curve.predict(Xva_struct_b)
        pred_rf_i_curve = rf_i_curve.predict(Xva_struct_i)

        # NB smoothing is also a hyperparameter; for curves, keep alpha=0.55 and nb_alpha=1.0
        nb_b_curve = MultinomialNB(alpha=1.0).fit(Xtr_text_b, ytr_b)
        nb_i_curve = MultinomialNB(alpha=1.0).fit(Xtr_text_i, ytr_i)
        nb_probs_b_curve = nb_b_curve.predict_proba(Xva_text_b)
        nb_probs_i_curve = nb_i_curve.predict_proba(Xva_text_i)

        pred_ens_b_curve = np.argmax(0.55 * nb_probs_b_curve + (1 - 0.55) * rf_probs_b_curve, axis=1)
        pred_ens_i_curve = np.argmax(0.55 * nb_probs_i_curve + (1 - 0.55) * rf_probs_i_curve, axis=1)

        hybrid_acc_b.append(float(accuracy_score(yva_b, pred_ens_b_curve)))
        rf_acc_b.append(float(accuracy_score(yva_b, pred_rf_b_curve)))
        hybrid_acc_i.append(float(accuracy_score(yva_i, pred_ens_i_curve)))
        rf_acc_i.append(float(accuracy_score(yva_i, pred_rf_i_curve)))

        # Full sweep for "best overall" settings (macro-F1)
        for nb_a in nb_smoothing:
            nb_b = MultinomialNB(alpha=nb_a).fit(Xtr_text_b, ytr_b)
            nb_i = MultinomialNB(alpha=nb_a).fit(Xtr_text_i, ytr_i)
            nb_probs_b = nb_b.predict_proba(Xva_text_b)
            nb_probs_i = nb_i.predict_proba(Xva_text_i)

            for mf in rf_max_features:
                for msl in rf_min_samples_leaf:
                    rf_b = RandomForestClassifier(
                        n_estimators=n,
                        random_state=0,
                        n_jobs=-1,
                        max_features=mf,
                        min_samples_leaf=msl,
                    ).fit(Xtr_struct_b, ytr_b)
                    rf_i = RandomForestClassifier(
                        n_estimators=n,
                        random_state=0,
                        n_jobs=-1,
                        max_features=mf,
                        min_samples_leaf=msl,
                    ).fit(Xtr_struct_i, ytr_i)

                    rf_probs_b = rf_b.predict_proba(Xva_struct_b)
                    rf_probs_i = rf_i.predict_proba(Xva_struct_i)

                    # pick best alpha for this (NB, RF) pair
                    for a in alphas:
                        pred_b = np.argmax(a * nb_probs_b + (1 - a) * rf_probs_b, axis=1)
                        f1_b = float(f1_score(yva_b, pred_b, average="macro", zero_division=0))
                        if f1_b > best_b["macro_f1"]:
                            best_b = {
                                "macro_f1": f1_b,
                                "n_estimators": n,
                                "alpha": a,
                                "nb_alpha": nb_a,
                                "rf_max_features": mf,
                                "rf_min_samples_leaf": msl,
                                "y_pred": pred_b,
                            }

                        pred_i = np.argmax(a * nb_probs_i + (1 - a) * rf_probs_i, axis=1)
                        f1_i = float(f1_score(yva_i, pred_i, average="macro", zero_division=0))
                        if f1_i > best_i["macro_f1"]:
                            best_i = {
                                "macro_f1": f1_i,
                                "n_estimators": n,
                                "alpha": a,
                                "nb_alpha": nb_a,
                                "rf_max_features": mf,
                                "rf_min_samples_leaf": msl,
                                "y_pred": pred_i,
                            }

        print(
            f"n={n} curves | baseline: ens_acc={hybrid_acc_b[-1]:.4f}, rf_acc={rf_acc_b[-1]:.4f} | "
            f"+interactions: ens_acc={hybrid_acc_i[-1]:.4f}, rf_acc={rf_acc_i[-1]:.4f}"
        )

    # Pick best n for comparisons
    # - Ensemble: choose best by macro-F1 (tracked in best_b/best_i)
    # - RF-only: choose best by accuracy curve (as before)
    best_n_h_b = int(best_b["n_estimators"])
    best_n_rf_b = n_estimators_list[int(np.argmax(rf_acc_b))]
    best_n_h_i = int(best_i["n_estimators"])
    best_n_rf_i = n_estimators_list[int(np.argmax(rf_acc_i))]

    pred_h_b = best_b["y_pred"]
    # Refit RF-only for the chosen curve best n (with representative params)
    rf_b_final = RandomForestClassifier(
        n_estimators=best_n_rf_b, random_state=0, n_jobs=-1, **curve_rf_kwargs
    ).fit(Xtr_struct_b, ytr_b)
    pred_rf_b = rf_b_final.predict(Xva_struct_b)
    pred_h_i = best_i["y_pred"]
    rf_i_final = RandomForestClassifier(
        n_estimators=best_n_rf_i, random_state=0, n_jobs=-1, **curve_rf_kwargs
    ).fit(Xtr_struct_i, ytr_i)
    pred_rf_i = rf_i_final.predict(Xva_struct_i)

    m_h_b = metrics(yva_b, pred_h_b)
    m_rf_b = metrics(yva_b, pred_rf_b)
    m_h_i = metrics(yva_i, pred_h_i)
    m_rf_i = metrics(yva_i, pred_rf_i)

    print("\n=== BASELINE (no interactions) ===")
    print(
        "Ensemble best: "
        f"n={best_n_h_b}, alpha={best_b['alpha']}, nb_alpha={best_b['nb_alpha']}, "
        f"rf_max_features={best_b['rf_max_features']}, rf_min_samples_leaf={best_b['rf_min_samples_leaf']}, "
        f"metrics={m_h_b}"
    )
    print(f"RF-only best_n={best_n_rf_b} (params={curve_rf_kwargs}), metrics={m_rf_b}")

    print("\n=== +INTERACTIONS ===")
    print(
        "Ensemble best: "
        f"n={best_n_h_i}, alpha={best_i['alpha']}, nb_alpha={best_i['nb_alpha']}, "
        f"rf_max_features={best_i['rf_max_features']}, rf_min_samples_leaf={best_i['rf_min_samples_leaf']}, "
        f"metrics={m_h_i}"
    )
    print(f"RF-only best_n={best_n_rf_i} (params={curve_rf_kwargs}), metrics={m_rf_i}")

    # Plot curves (validation accuracy)
    plt.figure(figsize=(8, 5))
    plt.plot(n_estimators_list, hybrid_acc_b, marker="o", label="Ensemble (baseline, best alpha per n)")
    plt.plot(n_estimators_list, rf_acc_b, marker="o", label="RF-only (baseline)")
    plt.plot(n_estimators_list, hybrid_acc_i, marker="o", label="Ensemble (+interactions, best alpha per n)")
    plt.plot(n_estimators_list, rf_acc_i, marker="o", label="RF-only (+interactions)")
    plt.xlabel("Random Forest n_estimators")
    plt.ylabel("Validation accuracy")
    plt.title("Hybrid vs RF-only (validation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nb_rf_ensemble_vs_rf_only_accuracy_curves.png", dpi=200)
    plt.show()

    # Confusion matrices (validation) + print matrices
    labels_b = unique_labels(yva_b, pred_h_b, pred_rf_b)
    cm_h_b = confusion_matrix(yva_b, pred_h_b, labels=labels_b)
    cm_rf_b = confusion_matrix(yva_b, pred_rf_b, labels=labels_b)

    labels_i = unique_labels(yva_i, pred_h_i, pred_rf_i)
    cm_h_i = confusion_matrix(yva_i, pred_h_i, labels=labels_i)
    cm_rf_i = confusion_matrix(yva_i, pred_rf_i, labels=labels_i)

    print("\n--- Confusion matrices (rows=true, cols=pred) ---")
    print(f"[baseline] Ensemble (n={best_n_h_b}, alpha={best_b['alpha']}):\n{cm_h_b}")
    print(f"[baseline] RF-only (n={best_n_rf_b}):\n{cm_rf_b}")
    print(f"[+interactions] Ensemble (n={best_n_h_i}, alpha={best_i['alpha']}):\n{cm_h_i}")
    print(f"[+interactions] RF-only (n={best_n_rf_i}):\n{cm_rf_i}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ConfusionMatrixDisplay(cm_h_b, display_labels=[str(x) for x in labels_b]).plot(
        ax=axes[0, 0], xticks_rotation=45, values_format="d"
    )
    axes[0, 0].set_title(
        f"Baseline Ensemble (n={best_n_h_b}, alpha={best_b['alpha']})\nmacroF1={m_h_b['macro_f1']:.4f}"
    )

    ConfusionMatrixDisplay(cm_rf_b, display_labels=[str(x) for x in labels_b]).plot(
        ax=axes[0, 1], xticks_rotation=45, values_format="d"
    )
    axes[0, 1].set_title(f"Baseline RF-only (n={best_n_rf_b})\nmacroF1={m_rf_b['macro_f1']:.4f}")

    ConfusionMatrixDisplay(cm_h_i, display_labels=[str(x) for x in labels_i]).plot(
        ax=axes[1, 0], xticks_rotation=45, values_format="d"
    )
    axes[1, 0].set_title(
        f"+Interactions Ensemble (n={best_n_h_i}, alpha={best_i['alpha']})\nmacroF1={m_h_i['macro_f1']:.4f}"
    )

    ConfusionMatrixDisplay(cm_rf_i, display_labels=[str(x) for x in labels_i]).plot(
        ax=axes[1, 1], xticks_rotation=45, values_format="d"
    )
    axes[1, 1].set_title(f"+Interactions RF-only (n={best_n_rf_i})\nmacroF1={m_rf_i['macro_f1']:.4f}")

    plt.tight_layout()
    plt.savefig("nb_rf_ensemble_vs_rf_only_confusion_matrices.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()