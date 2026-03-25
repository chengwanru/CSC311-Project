"""
naive_bayes.py
==============
Full Naive Bayes pipeline for CSC311 Project.
Reads from preprocessed.csv — basic cleaning in preprocessing.py;
train-fold median imputation and count upper caps in build_features().

Covers: feature engineering, model training,
        hyperparameter tuning, and evaluation.

Requires:  data_splitting.py in the same directory.
Run:       python naive_bayes.py
"""

import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
from data_splitting import grouped_kfold_split, regular_split

# ============================================================
# 0.  CONFIGURATION
# ============================================================

DATA_PATH   = "preprocessed.csv"
RANDOM_SEED = 108
K_FOLDS     = 5
USE_KFOLD   = True

# Hyper-parameter search grid
PARAM_GRID = {
    "vocab_size": [2000],
    "alpha":      [0.9],
    "binary_tf":  [True],   # binary presence vs raw TF
    "num_bins":   [None],    # None → Gaussian NB for numerics
    "blend_weight": [0.9, 1.0],
    "w_likert":      [1.0, 2.0],
    "w_numeric":     [0.5, 1.0],
    "w_price":       [0.0, 0.2],  # extra scale on price only (after w_numeric)
    "w_categorical": [1.0, 2.0],
}

# ============================================================
# 1.  COLUMN NAMES  (preprocessed short names)
# ============================================================

TARGET_COL = "Painting"
ID_COL     = "unique_id"

# Already parsed to integers 1–5 by preprocessing.py
LIKERT_COLS = ["sombre", "content", "calm", "uneasy"]

# emotion_intensity clipped in preprocessing; counts: lower bound only there,
# upper 99th-percentile cap fitted here on train (see build_features).
NUM_COLS  = ["emotion_intensity", "colours_noticed", "objects_noticed"]
PRICE_COL = "price_log1p"
# Count columns: train-only upper clip (mirrors old global 99th pct without leakage)
P99_UPPER_NUM_COLS = ["colours_noticed", "objects_noticed"]

# Still comma-separated strings → one-hot encoded here at model time
MULTI_COLS = ["room", "companion", "season"]

# Already cleaned strings → concatenated and vectorised here
TEXT_COLS = ["text_description", "text_food", "text_soundtrack"]

CLASS_NAMES = ["The Persistence of Memory", "The Starry Night", "The Water Lily Pond"]

# ============================================================
# 2.  MULTI-LABEL ENCODER  (only helper still needed)
# ============================================================

def encode_multi_label(series, known_values=None):
    """
    Expand a comma-separated column into binary indicator columns.
    known_values: sorted list built from the training fold only.
    Returns (encoded_df, known_values).
    """
    if known_values is None:
        label_set = set()
        for cell in series.dropna():
            for v in str(cell).split(","):
                label_set.add(v.strip())
        known_values = sorted(label_set)

    rows = []
    for cell in series:
        active = set()
        if not pd.isna(cell):
            for v in str(cell).split(","):
                active.add(v.strip())
        rows.append({v: int(v in active) for v in known_values})
    return pd.DataFrame(rows, index=series.index), known_values


# ============================================================
# 3.  TEXT → TERM FREQUENCY MATRIX
# ============================================================

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of",
    "is","it","this","that","with","as","was","are","be","i","me",
    "my","we","not","have","had","has","do","does","did","by","from",
    "its","into","so","if","no","up","out","about","which","like",
    "very","just","would","could","should","there","their","they",
    "when","what","how","all","some","any","can","more","been",
    "also","will","than","then","here","make","feel","feels","made",
    "makes",
    # domain stop words — appear in all classes equally
    "painting","art","picture","artwork","image","piece",
}


def stem(word):
    """
    Lightweight suffix-stripping stemmer (no external libraries).
    Handles the most common English inflections relevant to this dataset.
    Order matters — strip longer suffixes first.
    """
    # Protect very short words from over-stemming
    if len(word) <= 4:
        return word

    # Common verb / adjective suffixes
    for suffix in ("ingly", "ingly", "ness", "ment", "ful",
                   "less", "ing", "tion", "sion", "ous",
                   "ive", "ize", "ise", "est", "er", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: -len(suffix)]

    return word


def tokenize(text):
    """Lowercase, strip punctuation, remove stop words."""
    if not text or pd.isna(text):
        return [] 
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def build_vocabulary(texts, vocab_size):
    """Count word frequencies across all docs, return top-V words."""
    freq = defaultdict(int)
    for text in texts:
        for tok in tokenize(text):
            freq[tok] += 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])]
    return vocab[:vocab_size]


def texts_to_matrix(texts, vocab, binary=False):
    """
    Convert a list of strings into an (n, V) numpy matrix.
    binary=True → presence/absence; False → raw term counts.
    """
    word_index = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row_i, text in enumerate(texts):
        for tok in tokenize(text):
            if tok in word_index:
                mat[row_i, word_index[tok]] += 1
    if binary:
        mat = (mat > 0).astype(np.float32)
    return mat


# ============================================================
# 4.  FEATURE PIPELINE
# ============================================================

def build_features(df, fit_info=None, params=None):
    """
    Transform a preprocessed dataframe into a feature matrix.

    fit_info : dict from a previous fit call (pass when transforming
               val/test to avoid leakage). None means fit from df.
               Stores vocab, likert/num medians, numeric_p99_upper for counts,
               multi-label vocabularies, and optional bin/Gaussian stats.
    params   : dict with keys vocab_size, binary_tf, num_bins.

    Returns  : (X: np.ndarray, fit_info: dict)

    Feature block layout (left → right in X):
      [text (vocab_size)] [likert (4)] [numeric+price (4)] [multi-label (varies)]
    """
    if params is None:
        params = {"vocab_size": 500, "binary_tf": True, "num_bins": 5}

    df   = df.copy()
    parts = []
    info  = {} if fit_info is None else fit_info

    # ── 4a. Text features ─────────────────────────────────────
    combined_text = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)

    if fit_info is None:
        info["vocab"] = build_vocabulary(combined_text, params["vocab_size"])

    text_mat = texts_to_matrix(combined_text, info["vocab"],
                               binary=params["binary_tf"])
    parts.append(text_mat)                                    # (n, vocab_size)

    # ── 4b. Likert features (already integers 1–5) ────────────
    likert_df = df[LIKERT_COLS].astype(float)
    if fit_info is None:
        info["likert_medians"] = likert_df.median()          # fitted on train only
    likert_df = likert_df.fillna(info["likert_medians"])
    parts.append(likert_df.values.astype(np.float32))                                  # (n, 4)

    # ── 4c. Numeric + price features (already clean floats) ───
    num_df = df[NUM_COLS + [PRICE_COL]].astype(float)

    if fit_info is None:
        info["numeric_p99_upper"] = {}
        for c in P99_UPPER_NUM_COLS:
            col_vals = num_df[c].to_numpy(dtype=float, copy=False)
            finite = col_vals[~np.isnan(col_vals)]
            if finite.size == 0:
                info["numeric_p99_upper"][c] = None
            else:
                info["numeric_p99_upper"][c] = float(
                    np.ceil(np.percentile(finite, 99))
                )
    p99_upper = info.get("numeric_p99_upper", {})
    for c in P99_UPPER_NUM_COLS:
        hi = p99_upper.get(c)
        if hi is not None:
            num_df[c] = num_df[c].clip(upper=hi)

    if fit_info is None:
        info["num_medians"] = num_df.median()                # fitted on train only
    num_df = num_df.fillna(info["num_medians"])
    num_df = num_df.astype(np.float32)

    if params["num_bins"] is not None:
        # Discretise into bins → feed into Multinomial NB
        n_bins = params["num_bins"]
        if fit_info is None:
            bin_edges = {}
            for col in num_df.columns:
                edges = np.linspace(num_df[col].min(),
                                    num_df[col].max() + 1e-9,
                                    n_bins + 1)
                bin_edges[col] = edges
            info["bin_edges"] = bin_edges

        binned = np.zeros_like(num_df.values, dtype=np.float32)
        for j, col in enumerate(num_df.columns):
            binned[:, j] = np.digitize(num_df[col].values,
                                        info["bin_edges"][col][:-1]) - 1
        parts.append(binned)
    else:
        # Keep as continuous → Gaussian NB handles this block
        info["num_gaussian"] = True
        num_array = num_df.values.astype(np.float32)
        if fit_info is None:
            info["num_mean"] = num_array.mean(axis=0)
            info["num_std"]  = num_array.std(axis=0) + 1e-9
        num_array = (num_array - info["num_mean"]) / info["num_std"]
        parts.append(num_array)

    # ── 4d. Multi-label categorical features → one-hot ────────
    multi_parts = []
    for col in MULTI_COLS:
        kv_key = f"multi_{col}"
        enc, kv = encode_multi_label(df[col],
                                     known_values=info.get(kv_key))
        if fit_info is None:
            info[kv_key] = kv
        multi_parts.append(enc.values.astype(np.float32))
    parts.append(np.hstack(multi_parts))

    X = np.hstack(parts)

    # Record where each block starts and ends in X
    n_text     = parts[0].shape[1]
    n_likert   = parts[1].shape[1]
    n_numeric  = parts[2].shape[1]
    n_cat      = parts[3].shape[1]

    info["block_cols"] = {
        "text"     : (0,                              n_text),
        "likert"   : (n_text,                         n_text + n_likert),
        "numeric"  : (n_text + n_likert,              n_text + n_likert + n_numeric),
        "cat"      : (n_text + n_likert + n_numeric,  n_text + n_likert + n_numeric + n_cat),
    }

    return X, info


def gaussian_col_indices(params):
    """
    Returns the column indices in X that correspond to the numeric+price
    block when num_bins is None (Gaussian NB mode).
    Layout: [vocab_size | 4 likert | 4 num+price | multi]
    """
    if params["num_bins"] is not None:
        return []
    start = params["vocab_size"] + len(LIKERT_COLS)   # text + likert
    n_num = len(NUM_COLS) + 1                          # +1 for price
    return list(range(start, start + n_num))


# ============================================================
# 5.  NAIVE BAYES CLASSIFIER
# ============================================================

class NaiveBayes:
    """
    Mixed Naive Bayes:
    - Multinomial for text, binned numeric, and categorical features
    - Gaussian for continuous numeric features (when num_bins=None)

    Parameters are plain numpy arrays — easily saved/loaded for pred.py
    without any ML library dependency.
    """

    def __init__(self, alpha=1.0, gaussian_cols=None):
        self.alpha         = alpha
        self.gaussian_cols = gaussian_cols if gaussian_cols else []

    def _split_X(self, X):
        all_cols = np.arange(X.shape[1])
        g = np.array(self.gaussian_cols, dtype=int)
        m = np.array([c for c in all_cols
                      if c not in set(self.gaussian_cols)], dtype=int)
        return X[:, m], X[:, g], m, g

    def fit(self, X, y):
        n_samples  = X.shape[0]
        classes    = np.unique(y)
        n_classes  = len(classes)
        self.classes_ = classes

        # Log class priors
        self.log_priors_ = np.array([
            np.log((y == c).sum() / n_samples) for c in classes
        ])

        # Multinomial log-likelihoods
        X_mn, X_g, mn_idx, g_idx = self._split_X(X)
        n_mn = X_mn.shape[1]
        self.log_likelihoods_ = np.zeros((n_classes, n_mn))
        for k, c in enumerate(classes):
            counts = X_mn[y == c].sum(axis=0) + self.alpha
            self.log_likelihoods_[k] = np.log(counts / counts.sum())

        # Gaussian parameters
        if len(g_idx) > 0:
            self.gauss_mean_ = np.zeros((n_classes, len(g_idx)))
            self.gauss_std_  = np.zeros((n_classes, len(g_idx)))
            for k, c in enumerate(classes):
                X_c = X_g[y == c]
                self.gauss_mean_[k] = X_c.mean(axis=0)
                self.gauss_std_[k]  = X_c.std(axis=0) + 1e-9
        else:
            self.gauss_mean_ = None
            self.gauss_std_  = None

        self._mn_idx = mn_idx
        self._g_idx  = g_idx
        return self

    def _log_gaussian(self, X_g):
        """Log-PDF of Gaussian for each class. Returns (n, n_classes)."""
        n   = X_g.shape[0]
        out = np.zeros((n, len(self.classes_)))
        for k in range(len(self.classes_)):
            mu  = self.gauss_mean_[k]
            sig = self.gauss_std_[k]
            out[:, k] = -0.5 * np.sum(
                np.log(2 * np.pi * sig**2) + ((X_g - mu) / sig)**2,
                axis=1
            )
        return out

    def predict_log_proba(self, X):
        X_mn, X_g, _, _ = self._split_X(X)
        log_lik = X_mn @ self.log_likelihoods_.T   # (n, k)
        if len(self._g_idx) > 0:
            log_lik += self._log_gaussian(X_g)
        return log_lik + self.log_priors_

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class CNaiveBayes:
    """
    Complement Naive Bayes. Identical interface to NaiveBayes
    so it can be used interchangeably in the blending step.
    """

    def __init__(self, alpha=1.0, gaussian_cols=None):
        self.alpha         = alpha
        self.gaussian_cols = gaussian_cols if gaussian_cols else []

    def _split_X(self, X):
        all_cols = np.arange(X.shape[1])
        g = np.array(self.gaussian_cols, dtype=int)
        m = np.array([c for c in all_cols
                      if c not in set(self.gaussian_cols)], dtype=int)
        return X[:, m], X[:, g], m, g

    def fit(self, X, y):
        n_samples     = X.shape[0]
        classes       = np.unique(y)
        n_classes     = len(classes)
        self.classes_ = classes

        self.log_priors_ = np.array([
            np.log((y == c).sum() / n_samples) for c in classes
        ])

        X_mn, X_g, mn_idx, g_idx = self._split_X(X)
        n_mn = X_mn.shape[1]
        self.log_likelihoods_ = np.zeros((n_classes, n_mn))

        for k, c in enumerate(classes):
            # ← only change: sum over complement (all rows NOT in class k)
            counts = X_mn[y != c].sum(axis=0) + self.alpha
            # ← negate so argmax still picks the best class
            self.log_likelihoods_[k] = -np.log(counts / counts.sum())

        if len(g_idx) > 0:
            self.gauss_mean_ = np.zeros((n_classes, len(g_idx)))
            self.gauss_std_  = np.zeros((n_classes, len(g_idx)))
            for k, c in enumerate(classes):
                X_c = X_g[y == c]
                self.gauss_mean_[k] = X_c.mean(axis=0)
                self.gauss_std_[k]  = X_c.std(axis=0) + 1e-9
        else:
            self.gauss_mean_ = None
            self.gauss_std_  = None

        self._mn_idx = mn_idx
        self._g_idx  = g_idx
        return self

    def _log_gaussian(self, X_g):
        n   = X_g.shape[0]
        out = np.zeros((n, len(self.classes_)))
        for k in range(len(self.classes_)):
            mu  = self.gauss_mean_[k]
            sig = self.gauss_std_[k]
            out[:, k] = -0.5 * np.sum(
                np.log(2 * np.pi * sig**2) + ((X_g - mu) / sig)**2,
                axis=1
            )
        return out

    def predict_log_proba(self, X):
        X_mn, X_g, _, _ = self._split_X(X)
        log_lik = X_mn @ self.log_likelihoods_.T
        if len(self._g_idx) > 0:
            log_lik += self._log_gaussian(X_g)
        return log_lik + self.log_priors_

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def blended_predict(X, nb_model, cnb_model, weight):
    """
    Blend log-posteriors from NB and CNB.
    weight=1.0 → pure NB, weight=0.0 → pure CNB.
    Scores are normalised per-model before blending to keep
    them on a comparable scale.
    """
    nb_scores  = nb_model.predict_log_proba(X)
    cnb_scores = cnb_model.predict_log_proba(X)

    # Normalise each model's scores row-wise (subtract row max)
    nb_scores  -= nb_scores.max(axis=1, keepdims=True)
    cnb_scores -= cnb_scores.max(axis=1, keepdims=True)

    blended = weight * nb_scores + (1 - weight) * cnb_scores
    return nb_model.classes_[np.argmax(blended, axis=1)]

# ============================================================
# 6.  EVALUATION METRICS
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def confusion_matrix(y_true, y_pred, classes):
    n  = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, tc in enumerate(classes):
        for j, pc in enumerate(classes):
            cm[i, j] = np.sum((y_true == tc) & (y_pred == pc))
    return cm


def print_confusion_matrix(cm, classes):
    short  = ["Persistence", "Starry Night", "Water Lily"]
    header = f"{'':>15}" + "".join(f"{s:>15}" for s in short)
    print(header)
    for i, label in enumerate(short):
        row = f"{label:>15}" + "".join(f"{cm[i,j]:>15}" for j in range(3))
        print(row)


# ============================================================
# 7.  CROSS-VALIDATED HYPERPARAMETER SEARCH
# ============================================================

def apply_weights(X, params, block_cols):
    """
    Scale each feature block by its weight. Text stays at 1.0.
    Numeric block is scaled by w_numeric; price column (last in numeric block)
    is multiplied by w_price on top (effective price weight = w_numeric * w_price).
    """
    X = X.copy().astype(np.float32)
    for block, w in [("likert",   params["w_likert"]),
                     ("numeric",  params["w_numeric"]),
                     ("cat",      params["w_categorical"])]:
        start, end = block_cols[block]
        X[:, start:end] *= w

    price_col = block_cols["numeric"][1] - 1
    w_price = params.get("w_price", 0.0)
    X[:, price_col] *= w_price

    return X


def cross_validate_params(df, params):
    """5-fold grouped CV for one hyper-parameter combination."""
    fold_accs, fold_f1s = [], []

    if USE_KFOLD:
        splits = grouped_kfold_split(df, k=K_FOLDS, random_state=RANDOM_SEED)
    else:
        train_df, val_df, _ = regular_split(df, random_state=RANDOM_SEED)
        splits = [(train_df, val_df, None)]   # single iteration

    for train_df, val_df, _ in splits:

        y_train = train_df[TARGET_COL].values
        y_val   = val_df[TARGET_COL].values

        X_train, fit_info = build_features(train_df, fit_info=None,     params=params)
        X_val,   _        = build_features(val_df,   fit_info=fit_info,  params=params)

        X_train = apply_weights(X_train, params, fit_info["block_cols"])
        X_val   = apply_weights(X_val,   params, fit_info["block_cols"])

        g_cols   = gaussian_col_indices(params)
        nb_model  = NaiveBayes(alpha=params["alpha"],  gaussian_cols=g_cols)
        cnb_model = CNaiveBayes(alpha=params["alpha"], gaussian_cols=g_cols)
        nb_model.fit(X_train,  y_train)
        cnb_model.fit(X_train, y_train)

        y_pred = blended_predict(X_val, nb_model, cnb_model, params["blend_weight"])
        fold_accs.append(accuracy(y_val, y_pred))
        fold_f1s.append(macro_f1(y_val, y_pred, CLASS_NAMES))

    return np.mean(fold_accs), np.mean(fold_f1s)


def grid_search(df):
    """Exhaustive grid search over PARAM_GRID."""
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    print(f"\n{'='*70}")
    print("HYPERPARAMETER GRID SEARCH")
    print(f"{'='*70}")
    print(f"{'vocab':>6} {'alpha':>6} {'binary':>7} {'bins':>5} {'blend':>5} "
          f"{'lik':>5} {'num':>5} {'price':>5} {'cat':>5} "
          f"{'Val Acc':>9} {'Val F1':>9}")
    print("-" * 76)

    best_f1, best_params, all_results = -1, None, []

    for combo in product(*values):
        params  = dict(zip(keys, combo))
        val_acc, val_f1 = cross_validate_params(df, params)

        b_str  = "T" if params["binary_tf"] else "F"
        bins_s = str(params["num_bins"]) if params["num_bins"] else "gauss"
        print(f"{params['vocab_size']:>6} {params['alpha']:>6.2f} {b_str:>7} "
              f"{bins_s:>5} {params['blend_weight']:>5} {params['w_likert']:>5}"
              f"{params['w_numeric']:>5} {params['w_price']:>5} {params['w_categorical']:>5} "
              f"{val_acc:>9.4f} {val_f1:>9.4f}")

        all_results.append((params, val_acc, val_f1))
        if val_f1 > best_f1:
            best_f1, best_params = val_f1, params

    print("-" * 76)
    print(f"\n✓ Best params  : {best_params}")
    print(f"  Best Val F1  : {best_f1:.4f}\n")
    return best_params, all_results


# ============================================================
# 8.  FINAL EVALUATION ON HELD-OUT TEST SET
# ============================================================

def final_evaluation(df, best_params):
    """
    Train on the full 80% train+val pool, evaluate on the fixed
    20% held-out test set (same split used across all folds).
    """
    print(f"{'='*70}")
    print("FINAL EVALUATION — HELD-OUT TEST SET")
    print(f"{'='*70}")

    # Collect the fixed test set and all trainval IDs
    test_df      = None
    trainval_ids = []
    for train_df, _, t_df in grouped_kfold_split(
            df, k=K_FOLDS, random_state=RANDOM_SEED):
        if test_df is None:
            test_df = t_df
        trainval_ids.extend(train_df[ID_COL].values)

    trainval_ids = np.unique(trainval_ids)
    trainval_df  = df[df[ID_COL].isin(trainval_ids)]

    X_train, fit_info = build_features(trainval_df, fit_info=None,    params=best_params)
    X_test,  _        = build_features(test_df,     fit_info=fit_info, params=best_params)

    X_train = apply_weights(X_train, best_params, fit_info["block_cols"])
    X_test  = apply_weights(X_test,  best_params, fit_info["block_cols"])

    y_train = trainval_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    g_cols    = gaussian_col_indices(best_params)
    nb_model  = NaiveBayes(alpha=best_params["alpha"],  gaussian_cols=g_cols)
    cnb_model = CNaiveBayes(alpha=best_params["alpha"], gaussian_cols=g_cols)
    nb_model.fit(X_train,  y_train)
    cnb_model.fit(X_train, y_train)

    y_pred = blended_predict(X_test, nb_model, cnb_model, best_params["blend_weight"])

    test_acc = accuracy(y_test, y_pred)
    test_f1  = macro_f1(y_test, y_pred, CLASS_NAMES)
    cm       = confusion_matrix(y_test, y_pred, CLASS_NAMES)

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test Macro-F1 : {test_f1:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print_confusion_matrix(cm, CLASS_NAMES)

    # print(f"\nMISCLASSIFIED ENTRIES ({np.sum(y_pred != y_test)} total)")
    # print("-" * 70)

    # short = {
    #     "The Persistence of Memory": "Persistence",
    #     "The Starry Night":          "Starry Night",
    #     "The Water Lily Pond":       "Water Lily",
    # }

    # misclassified = test_df[y_pred != y_test].copy()
    # misclassified["predicted"] = y_pred[y_pred != y_test]

    # for _, row in misclassified.iterrows():
    #     true_label = short[row[TARGET_COL]]
    #     pred_label = short[row["predicted"]]
    #     print(f"\n  ID {int(row[ID_COL])}  |  True: {true_label:<15}  Predicted: {pred_label}")
    #     print(f"  Description : {str(row['text_description'])[:120]}")
    #     print(f"  Food        : {str(row['text_food'])[:60]}")
    #     print(f"  Soundtrack  : {str(row['text_soundtrack'])[:80]}")
    #     print(f"  Sombre={row['sombre']}  Content={row['content']}  "
    #         f"Calm={row['calm']}  Uneasy={row['uneasy']}  "
    #         f"Emotion={row['emotion_intensity']}")

    return nb_model, cnb_model, fit_info, test_acc, test_f1


# ============================================================
# 9.  SAVE MODEL PARAMETERS (for pred.py)
# ============================================================

def save_model(nb_model, cnb_model, fit_info, best_params, out_dir="./"):
    """
    Export learned parameters as .npy + JSON so that pred.py can run
    inference using only numpy and pandas — no sklearn required.
    """
    np.save(f"{out_dir}nb_log_priors.npy",     nb_model.log_priors_)
    np.save(f"{out_dir}nb_log_likelihoods.npy", nb_model.log_likelihoods_)
    np.save(f"{out_dir}nb_mn_idx.npy",          nb_model._mn_idx)
    np.save(f"{out_dir}nb_g_idx.npy",           nb_model._g_idx)
    np.save(f"{out_dir}nb_classes.npy",         nb_model.classes_)

    if nb_model.gauss_mean_ is not None:
        np.save(f"{out_dir}nb_gauss_mean.npy", nb_model.gauss_mean_)
        np.save(f"{out_dir}nb_gauss_std.npy",  nb_model.gauss_std_)

    np.save(f"{out_dir}cnb_log_priors.npy",      cnb_model.log_priors_)
    np.save(f"{out_dir}cnb_log_likelihoods.npy",  cnb_model.log_likelihoods_)
    np.save(f"{out_dir}cnb_mn_idx.npy",           cnb_model._mn_idx)
    np.save(f"{out_dir}cnb_g_idx.npy",            cnb_model._g_idx)
    np.save(f"{out_dir}cnb_classes.npy",          cnb_model.classes_)

    if cnb_model.gauss_mean_ is not None:
        np.save(f"{out_dir}cnb_gauss_mean.npy", cnb_model.gauss_mean_)
        np.save(f"{out_dir}cnb_gauss_std.npy",  cnb_model.gauss_std_)

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):    return obj.tolist()
        if isinstance(obj, np.integer):    return int(obj)
        if isinstance(obj, np.floating):   return float(obj)
        if isinstance(obj, pd.Series):     return obj.to_dict()
        if isinstance(obj, dict):          return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):          return [to_serializable(i) for i in obj]
        return obj

    with open(f"{out_dir}nb_fit_info.json", "w") as f:
        json.dump(to_serializable(fit_info), f, indent=2)

    with open(f"{out_dir}nb_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n✓ Model saved to {out_dir}")
    print("  Files: nb_log_priors.npy, nb_log_likelihoods.npy,")
    print("         nb_mn_idx.npy, nb_g_idx.npy, nb_classes.npy,")
    print("         nb_fit_info.json, nb_params.json")


# ============================================================
# 10.  MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows | {df[ID_COL].nunique()} students | "
          f"{df[TARGET_COL].nunique()} classes\n")

    best_params, all_results = grid_search(df)
    nb_model, cnb_model, fit_info, test_acc, test_f1 = final_evaluation(df, best_params)
    save_model(nb_model, cnb_model, fit_info, best_params)
