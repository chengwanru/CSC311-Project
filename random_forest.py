import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from feature_names import feature_names_from_state

# Load preprocessed arrays
data = np.load("preprocessed_arrays.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# Vary number of trees
tree_counts = list(range(10, 201, 10))
val_acc = []

for n in tree_counts:
    rf = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_acc.append(rf.score(X_val, y_val))
    print(f"n_estimators={n}, val_acc={val_acc[-1]:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(tree_counts, val_acc, marker="o")
plt.xlabel("Number of Trees")
plt.ylabel("Validation Accuracy")
plt.title("Random Forest Performance vs Number of Trees")
plt.grid(True)
plt.tight_layout()
plt.show()

# Best model by validation accuracy — metrics and trees use validation only (no test set)
best_n = tree_counts[int(np.argmax(val_acc))]
print(f"\nBest n_estimators (by validation accuracy): {best_n}")

rf_best = RandomForestClassifier(n_estimators=best_n, random_state=0, n_jobs=-1)
rf_best.fit(X_train, y_train)
y_val_pred = rf_best.predict(X_val)

label_order = unique_labels(y_val, y_val_pred)
cm = confusion_matrix(y_val, y_val_pred, labels=label_order)
macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

print("\n--- Validation confusion matrix (rows=true, cols=pred) ---")
print(cm)
print(f"\nMacro-averaged F1 (validation): {macro_f1:.4f}")

fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[str(lab) for lab in label_order],
)
disp.plot(ax=ax_cm, xticks_rotation=45, values_format="d")
ax_cm.set_title(f"Validation confusion matrix (n_estimators={best_n})")
plt.tight_layout()
plt.show()

# A few individual trees (structure truncated for readability)
with open("preprocess_state.pkl", "rb") as f:
    _state = pickle.load(f)
feature_names = feature_names_from_state(_state)
if len(feature_names) != X_train.shape[1]:
    raise ValueError(
        f"Feature name count {len(feature_names)} != X columns {X_train.shape[1]}; "
        "re-run run_preprocessing.py so preprocess_state.pkl matches preprocessed_arrays.npz."
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
    ax.set_title(f"Estimator {i + 1} / {best_n} (first {max_depth_draw} levels)")
fig_trees.suptitle("Sample trees from the fitted random forest", y=1.02)
plt.tight_layout()
plt.show()
