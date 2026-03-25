import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# Load preprocessed arrays
# -------------------------------
data = np.load("preprocessed_arrays.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# -------------------------------
# Load TF-IDF vectorizer to get number of text features
# -------------------------------
with open("preprocess_state.pkl", "rb") as f:
    state = pickle.load(f)

n_text_features = state["_vectorizer"].transform(["test"]).shape[1]
n_total_features = X_train.shape[1]
n_structured_features = n_total_features - n_text_features

# -------------------------------
# Split features
# -------------------------------
X_train_struct = X_train[:, :n_structured_features]
X_val_struct = X_val[:, :n_structured_features]

X_train_text = X_train[:, n_structured_features:]
X_val_text = X_val[:, n_structured_features:]

# Ensure non-negative for MultinomialNB
X_train_text = np.maximum(X_train_text, 0)
X_val_text = np.maximum(X_val_text, 0)

# -------------------------------
# Train Naive Bayes on text
# -------------------------------
nb = MultinomialNB()
nb.fit(X_train_text, y_train)
nb_val_probs = nb.predict_proba(X_val_text)

# -------------------------------
# Train Logistic Regression on structured features
# -------------------------------
logreg = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=500
)
logreg.fit(X_train_struct, y_train)
logreg_val_probs = logreg.predict_proba(X_val_struct)

# -------------------------------
# Combine probabilities (average)
# -------------------------------
hybrid_val_probs = (nb_val_probs + logreg_val_probs) / 2
y_val_pred = np.argmax(hybrid_val_probs, axis=1)

acc = accuracy_score(y_val, y_val_pred)
print(f"Hybrid Naive Bayes + Logistic Regression validation accuracy: {acc:.4f}")

# -------------------------------
# Optional: test effect of regularization on logistic regression
# -------------------------------
C_list = [0.01, 0.1, 1, 10, 100]
acc_list = []

for C in C_list:
    logreg = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', C=C, max_iter=500
    )
    logreg.fit(X_train_struct, y_train)
    logreg_val_probs = logreg.predict_proba(X_val_struct)
    hybrid_val_probs = (nb_val_probs + logreg_val_probs) / 2
    y_val_pred = np.argmax(hybrid_val_probs, axis=1)
    acc_list.append(accuracy_score(y_val, y_val_pred))

# Plot effect of regularization
plt.figure(figsize=(6,4))
plt.plot(C_list, acc_list, marker='o')
plt.xscale('log')
plt.xlabel("Logistic Regression C (inverse regularization)")
plt.ylabel("Hybrid Model Validation Accuracy")
plt.title("Effect of Logistic Regression Regularization in Hybrid Model")
plt.grid(True)
plt.show()