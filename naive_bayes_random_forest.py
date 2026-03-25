import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.inspection import PartialDependenceDisplay
import graphviz

# -------------------------------
# Load preprocessed arrays
# -------------------------------
data = np.load("preprocessed_arrays.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# -------------------------------
# Load TF-IDF vectorizer state
# -------------------------------
with open("preprocess_state.pkl", "rb") as f:
    state = pickle.load(f)

n_text_features = state["_vectorizer"].transform(["test"]).shape[1]
n_total_features = X_train.shape[1]
n_other_features = n_total_features - n_text_features

# -------------------------------
# Split features
# -------------------------------
X_train_other = X_train[:, :n_other_features]
X_val_other = X_val[:, :n_other_features]

X_train_text = X_train[:, n_other_features:]
X_val_text = X_val[:, n_other_features:]

# Ensure non-negative for MultinomialNB
X_train_text = np.maximum(X_train_text, 0)
X_val_text = np.maximum(X_val_text, 0)

# -------------------------------
# Naive Bayes on TF-IDF text
# -------------------------------
nb = MultinomialNB()
nb.fit(X_train_text, y_train)
nb_val_probs = nb.predict_proba(X_val_text)

# -------------------------------
# Random Forest with varying n_estimators
# -------------------------------
n_estimators_list = [10, 50, 100, 150, 200, 250]
val_acc_list = []

for n_estimators in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    rf.fit(X_train_other, y_train)
    rf_val_probs = rf.predict_proba(X_val_other)

    # Hybrid probabilities
    combined_probs = (nb_val_probs + rf_val_probs) / 2
    y_val_pred = np.argmax(combined_probs, axis=1)
    acc = accuracy_score(y_val, y_val_pred)
    val_acc_list.append(acc)
    print(f"n_estimators={n_estimators}, hybrid validation accuracy={acc:.4f}")

# -------------------------------
# Plot hybrid performance
# -------------------------------
plt.figure(figsize=(6,4))
plt.plot(n_estimators_list, val_acc_list, marker='o')
plt.xlabel("Random Forest n_estimators")
plt.ylabel("Hybrid Model Validation Accuracy")
plt.title("Hybrid Naive Bayes + Random Forest Performance")
plt.grid(True)
plt.show()

# -------------------------------
# Feature importance from RF (structured features only)
# -------------------------------
feature_names_struct = [
    "Emotion", "N_colours", "N_objects", "Sombre", "Content", "Calm", "Uneasy", "Price"
]
importances = rf.feature_importances_[:len(feature_names_struct)]
plt.figure(figsize=(6,4))
plt.barh(feature_names_struct, importances)
plt.xlabel("Random Forest Feature Importance")
plt.title("Structured Feature Importance")
plt.show()

# -------------------------------
# Visualize a single tree from RF
# -------------------------------
single_tree = rf.estimators_[0]
dot_data = tree.export_graphviz(
    single_tree,
    feature_names=[f"F{i}" for i in range(X_train_other.shape[1])],  # avoid mismatch
    class_names=list(state["class_to_idx"].keys()),
    filled=True,
    rounded=True,
    max_depth=4
)
graph = graphviz.Source(dot_data)
graph.render("single_tree_example")  # saves PDF
graph.view()  # opens default viewer

# -------------------------------
# Partial Dependence Plots (choose one painting as target)
# -------------------------------
# Example: "The Water Lily Pond"
target_class = state["class_to_idx"]["The Water Lily Pond"]
key_features_indices = [0, 1, 2, 3]  # first 4 structured features
PartialDependenceDisplay.from_estimator(
    rf,
    X_val_other,
    features=key_features_indices,
    kind="average",
    target=target_class
)
plt.suptitle("Partial Dependence of Structured Features for Lily Pond")
plt.show()

# -------------------------------
# Predicted probabilities heatmap
# -------------------------------
probs_df = pd.DataFrame(combined_probs, columns=list(state["class_to_idx"].keys()))
probs_df["true"] = [list(state["class_to_idx"].keys())[y] for y in y_val]

plt.figure(figsize=(10,6))
sns.heatmap(probs_df.iloc[:50, :-1], cmap="viridis", annot=True)
plt.title("Hybrid Model Predicted Probabilities (first 50 validation samples)")
plt.ylabel("Sample index")
plt.xlabel("Painting class")
plt.show()