import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from feature_names import feature_names_from_state

BASELINE_ARRAYS = "preprocessed_arrays.npz"
INTER_ARRAYS = "preprocessed_arrays_interactions.npz"

with open("preprocess_state.pkl", "rb") as f:
    _state = pickle.load(f)

def _load_npz(path):
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"]

X_train_b, y_train_b, X_val_b, y_val_b = _load_npz(BASELINE_ARRAYS)
X_train_i, y_train_i, X_val_i, y_val_i = _load_npz(INTER_ARRAYS)

# Vary number of trees
tree_counts = list(range(10, 201, 10))
val_acc_b = []
val_acc_i = []

for n in tree_counts:
    rf_b = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    rf_b.fit(X_train_b, y_train_b)
    val_acc_b.append(rf_b.score(X_val_b, y_val_b))

    rf_i = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    rf_i.fit(X_train_i, y_train_i)
    val_acc_i.append(rf_i.score(X_val_i, y_val_i))

    print(
        f"n_estimators={n}, "
        f"baseline_val_acc={val_acc_b[-1]:.4f}, "
        f"+interactions_val_acc={val_acc_i[-1]:.4f}"
    )

# Plot
plt.figure(figsize=(8, 5))
plt.plot(tree_counts, val_acc_b, marker="o", label="baseline")
plt.plot(tree_counts, val_acc_i, marker="o", label="+ interactions")
plt.xlabel("Number of Trees")
plt.ylabel("Validation Accuracy")
plt.title("Random Forest Performance vs Number of Trees (validation)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def eval_variant(name, X_train, y_train, X_val, y_val, add_interactions):
    best_n = tree_counts[int(np.argmax(val_acc_i if add_interactions else val_acc_b))]
    rf = RandomForestClassifier(n_estimators=best_n, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_val_pred = rf.predict(X_val)

    label_order = unique_labels(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred, labels=label_order)
    macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

    print(f"\n[{name}] best_n_estimators={best_n}")
    print(f"[{name}] macro_f1_val={macro_f1:.4f}")
    print(f"[{name}] confusion_matrix_val:\n{cm}")

    return best_n, rf, y_val_pred, label_order, cm, macro_f1

best_n_b, rf_b, y_pred_b, labels_b, cm_b, f1_b = eval_variant(
    "baseline", X_train_b, y_train_b, X_val_b, y_val_b, add_interactions=False
)
best_n_i, rf_i, y_pred_i, labels_i, cm_i, f1_i = eval_variant(
    "+interactions", X_train_i, y_train_i, X_val_i, y_val_i, add_interactions=True
)

fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
ConfusionMatrixDisplay(cm_b, display_labels=[str(x) for x in labels_b]).plot(
    ax=ax1, xticks_rotation=45, values_format="d"
)
ax1.set_title(f"Baseline (n_estimators={best_n_b})")
ConfusionMatrixDisplay(cm_i, display_labels=[str(x) for x in labels_i]).plot(
    ax=ax2, xticks_rotation=45, values_format="d"
)
ax2.set_title(f"+ interactions (n_estimators={best_n_i})")
plt.suptitle("Validation confusion matrices")
plt.tight_layout()
plt.show()

# Visualize trees for the better-performing variant (by macro-F1 on validation)
use_inter = f1_i >= f1_b
rf_best = rf_i if use_inter else rf_b
X_train = X_train_i if use_inter else X_train_b
variant_name = "+ interactions" if use_inter else "baseline"
best_n = best_n_i if use_inter else best_n_b

feature_names = feature_names_from_state(_state, add_interactions=use_inter)
if len(feature_names) != X_train.shape[1]:
    raise ValueError(
        f"Feature name count {len(feature_names)} != X columns {X_train.shape[1]}; "
        "re-run run_preprocessing.py so preprocess_state.pkl matches the arrays."
    )

idx_to_painting = {i: name for name, i in _state["class_to_idx"].items()}
tree0 = rf_best.estimators_[0]
class_names = [idx_to_painting[int(c)] for c in tree0.classes_]
max_depth_draw = 4

fig_trees, axes = plt.subplots(1, 3, figsize=(26, 9), dpi=90)
for i, ax in enumerate(axes):
    plot_tree(
        rf_best.estimators_[i],
        ax=ax,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=5,
        max_depth=max_depth_draw,
    )
    ax.set_title(f"{variant_name}: estimator {i + 1} / {best_n} (first {max_depth_draw} levels)")
fig_trees.suptitle("Sample trees from the fitted random forest", y=1.02)
plt.tight_layout()
plt.show()
