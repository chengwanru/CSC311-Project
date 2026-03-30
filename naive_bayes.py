import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB


def top_tokens_gaussian_on_tfidf(gnb: GaussianNB, tokens: np.ndarray, top_k: int = 20):
    """
    Use standardized mean differences on TF-IDF dimensions only:
      score_c[j] = (mu_c[j] - mean_{c'!=c} mu_{c'}[j]) / sqrt(var_c[j] + mean_{c'!=c} var_{c'}[j])
    Returns dict[class_index] -> list[(token, score)].
    """
    mu = gnb.theta_  # [C, D]
    var = gnb.var_   # [C, D]

    d_text = len(tokens)
    d_total = mu.shape[1]
    d_other = d_total - d_text

    mu_t = mu[:, d_other:]
    var_t = var[:, d_other:]

    out = {}
    for c in range(mu_t.shape[0]):
        mu_others = np.delete(mu_t, c, axis=0)
        var_others = np.delete(var_t, c, axis=0)
        mu_o = mu_others.mean(axis=0)
        var_o = var_others.mean(axis=0)
        denom = np.sqrt(var_t[c] + var_o + 1e-12)
        score = (mu_t[c] - mu_o) / denom
        idx = np.argsort(np.abs(score))[-top_k:][::-1]
        out[c] = [(str(tokens[i]), float(score[i])) for i in idx]
    return out


def main():
    with open("preprocess_state.pkl", "rb") as f:
        state = pickle.load(f)

    data = np.load("preprocessed_arrays.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    vocab = state["_vectorizer"].get_feature_names_out()

    # GaussianNB handles continuous + possibly negative features (your numeric are z-scored)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    idx_to_painting = {i: name for name, i in state["class_to_idx"].items()}

    print("Naive Bayes (GaussianNB on ALL features: structured + TF-IDF)")
    print(f"val_acc: {acc:.4f}")
    print(f"val_macro_f1: {macro_f1:.4f}")

    top = top_tokens_gaussian_on_tfidf(nb, vocab, top_k=20)
    print("\nTop TF-IDF tokens by class (largest |standardized mean diff|):")
    for c in nb.classes_:
        print(f"\nClass {int(c)} = {idx_to_painting[int(c)]}")
        for tok, score in top[int(c)]:
            print(f"  {tok:>20s}  {score:+.3f}")


if __name__ == "__main__":
    main()

