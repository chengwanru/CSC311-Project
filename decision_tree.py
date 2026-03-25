import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# -------------------------------
# Load preprocessed arrays
# -------------------------------
data = np.load("preprocessed_arrays.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# -------------------------------
# Single Decision Tree Baseline
# -------------------------------
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
dt_train_acc = dt.score(X_train, y_train)
dt_val_acc = dt.score(X_val, y_val)

print(f"Decision Tree baseline:")
print(f"  Train accuracy: {dt_train_acc:.4f}")
print(f"  Validation accuracy: {dt_val_acc:.4f}")

# -------------------------------
# Random Forest with varying trees
# -------------------------------
tree_counts = range(10, 201, 10)  # 10 to 200 trees in steps of 10
rf_val_acc = []

for n in tree_counts:
    rf = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_acc.append(rf.score(X_val, y_val))
    print(f"n_estimators={n}, val_acc={rf_val_acc[-1]:.4f}")

# -------------------------------
# Plot Random Forest performance
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(tree_counts, rf_val_acc, marker="o", label="Random Forest")
plt.axhline(y=dt_val_acc, color='r', linestyle='--', label="Decision Tree baseline")
plt.xlabel("Number of Trees")
plt.ylabel("Validation Accuracy")
plt.title("Random Forest vs Single Decision Tree")
plt.grid(True)
plt.legend()
plt.show()