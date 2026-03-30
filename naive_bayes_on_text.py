import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB


def top_tokens_multinomial(nb: MultinomialNB, tokens: np.ndarray, top_k: int = 20):
    """
    For each class c, rank tokens by log P(token|c) - mean_{c'!=c} log P(token|c').
    Returns dict[class_index] -> list[(token, score)].
    """
    logp = nb.feature_log_prob_  # shape [C, V]
    out = {}
    for c in range(logp.shape[0]):
        others = np.delete(logp, c, axis=0)
        score = logp[c] - others.mean(axis=0)
        idx = np.argsort(score)[-top_k:][::-1]
        out[c] = [(str(tokens[i]), float(score[i])) for i in idx]
    return out


def main():
    with open("preprocess_state.pkl", "rb") as f:
        state = pickle.load(f)

    data = np.load("preprocessed_arrays.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    vocab = state["_vectorizer"].get_feature_names_out()
    n_text = len(vocab)
    n_other = X_train.shape[1] - n_text

    X_train_text = np.maximum(X_train[:, n_other:], 0)
    X_val_text = np.maximum(X_val[:, n_other:], 0)

    nb = MultinomialNB()
    nb.fit(X_train_text, y_train)
    y_pred = nb.predict(X_val_text)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    idx_to_painting = {i: name for name, i in state["class_to_idx"].items()}

    print("Naive Bayes (text-only TF-IDF)")
    print(f"val_acc: {acc:.4f}")
    print(f"val_macro_f1: {macro_f1:.4f}")

    top = top_tokens_multinomial(nb, vocab, top_k=20)
    print("\nTop tokens by class (most discriminative for that class):")
    for c in nb.classes_:
        print(f"\nClass {int(c)} = {idx_to_painting[int(c)]}")
        for tok, score in top[int(c)]:  # classes are 0..K-1 here
            print(f"  {tok:>20s}  {score:+.3f}")


if __name__ == "__main__":
    main()

