"""
data_exploration_histograms.py
================================
Generates histograms for all non-text features, broken down by painting.

Output structure:
  histograms/
  ├── The Persistence of Memory/
  │   ├── emotional_intensity.png
  │   ├── colours_noticed.png
  │   └── ...
  ├── The Starry Night/
  │   └── ...
  └── The Water Lily Pond/
      └── ...

Each painting folder contains one histogram per non-text feature,
showing that painting's distribution overlaid against the full dataset.

Run:  python data_exploration_histograms.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH  = "preprocessed.csv"
OUTPUT_DIR = "histograms"

PAINTINGS = [
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
]

# Colour palette: one per painting (muted but distinct)
PAINTING_COLORS = {
    "The Persistence of Memory": "#E07B54",   # warm orange-brown
    "The Starry Night":          "#4A7FB5",   # deep blue
    "The Water Lily Pond":       "#5BAD72",   # soft green
}

# ── Non-text feature definitions ─────────────────────────────
# Each entry: (column_name, short_label, feature_type, x_label)
# feature_type: "numeric" | "likert" | "price" | "multi_category"
#
# Data is already cleaned by preprocessing.py:
#   - Likert columns are integers 1–5 (no parsing needed)
#   - Numeric columns are clipped floats (no parsing needed)
#   - Price is already log1p-transformed (no parsing needed)
#   - Category columns are comma-separated strings (no parsing needed)

FEATURES = [
    (
        "emotion_intensity",
        "emotion_intensity",
        "numeric",
        "Emotional Intensity (1–10)",
    ),
    (
        "sombre",
        "sombre",
        "likert",
        "Sombre (1=Strongly Disagree → 5=Strongly Agree)",
    ),
    (
        "content",
        "content",
        "likert",
        "Content (1=Strongly Disagree → 5=Strongly Agree)",
    ),
    (
        "calm",
        "calm",
        "likert",
        "Calm (1=Strongly Disagree → 5=Strongly Agree)",
    ),
    (
        "uneasy",
        "uneasy",
        "likert",
        "Uneasy (1=Strongly Disagree → 5=Strongly Agree)",
    ),
    (
        "colours_noticed",
        "colours_noticed",
        "numeric",
        "Number of Prominent Colours Noticed",
    ),
    (
        "objects_noticed",
        "objects_noticed",
        "numeric",
        "Number of Objects Noticed",
    ),
    (
        "price_log1p",
        "price_log1p",
        "price",
        "Willingness to Pay (CAD, log scale)",
    ),
    (
        "room",
        "room",
        "multi_category",
        "Room Preference",
    ),
    (
        "companion",
        "companion",
        "multi_category",
        "Preferred Viewing Companion",
    ),
    (
        "season",
        "season",
        "multi_category",
        "Season Association",
    ),
]

# ============================================================
# HELPERS
# ============================================================

def expand_multi_label(series):
    """
    'Living room,Bedroom' → ['Living room', 'Bedroom'].
    Returns a flat list of atomic labels across all non-null rows.
    """
    labels = []
    for cell in series.dropna():
        for v in str(cell).split(","):
            v = v.strip()
            if v:
                labels.append(v)
    return labels


# ============================================================
# PLOT HELPERS
# ============================================================

def _style_ax(ax, title, xlabel, ylabel="Count"):
    """Consistent axis styling across all plots."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def plot_numeric_histogram(painting, values_painting, xlabel, title, save_path, bins=15):
    """Histogram of a continuous numeric column for one painting."""
    color = PAINTING_COLORS[painting]
    vals  = values_painting[~np.isnan(values_painting)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bin_edges = np.linspace(vals.min(), vals.max(), bins + 1)
    ax.hist(vals, bins=bin_edges, color=color, edgecolor="white")

    med = np.median(vals)
    ax.axvline(med, color="#333333", linestyle="--", linewidth=1.4,
               label=f"Median = {med:.1f}")
    ax.legend(fontsize=9, framealpha=0.6)

    _style_ax(ax, title, xlabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_likert_histogram(painting, values_painting, xlabel, title, save_path):
    """Bar chart of raw counts for each Likert level (1–5)."""
    color = PAINTING_COLORS[painting]
    levels     = [1, 2, 3, 4, 5]
    labels_map = {1: "1\nStr. Disagree", 2: "2\nDisagree",
                  3: "3\nNeutral",       4: "4\nAgree",
                  5: "5\nStr. Agree"}

    counts = [int(np.sum(values_painting == lv)) for lv in levels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(len(levels)), counts, color=color, edgecolor="white", width=0.6)
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([labels_map[lv] for lv in levels], fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    _style_ax(ax, title, xlabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_price_histogram(painting, log_prices_painting, title, save_path):
    """
    Count histogram of the already log1p-transformed price column.
    X-axis tick labels show the original CAD values for readability.
    """
    color = PAINTING_COLORS[painting]
    vals  = log_prices_painting[~np.isnan(log_prices_painting)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bin_edges = np.linspace(vals.min(), vals.max(), 16)
    ax.hist(vals, bins=bin_edges, color=color, edgecolor="white")

    med = np.median(vals)
    ax.axvline(med, color="#333333", linestyle="--", linewidth=1.4,
               label=f"Median = ${np.expm1(med):.0f}")
    ax.legend(fontsize=9, framealpha=0.6)

    # Remap x-axis to show original CAD values
    tick_raw = [0, 10, 50, 100, 500, 1000, 5000, 10_000]
    ax.set_xticks([np.log1p(v) for v in tick_raw])
    ax.set_xticklabels([f"${v:,.0f}" for v in tick_raw],
                       rotation=35, ha="right", fontsize=9)

    _style_ax(ax, title, "Willingness to Pay (CAD)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_category_histogram(painting, labels_painting, xlabel, title, save_path):
    """
    Vertical bar chart of raw counts per category for one painting.
    Sorted by frequency, most common on the left.
    """
    from collections import Counter
    color = PAINTING_COLORS[painting]

    counts = Counter(labels_painting)
    cats   = [c for c, _ in counts.most_common()]
    values = [counts[c] for c in cats]

    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 0.75), 4.5))
    ax.bar(np.arange(len(cats)), values, color=color, edgecolor="white", width=0.6)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    _style_ax(ax, title, xlabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading data from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows loaded\n")

    # ── Create output folders ─────────────────────────────────
    for painting in PAINTINGS:
        os.makedirs(os.path.join(OUTPUT_DIR, painting), exist_ok=True)

    print("Generating histograms...\n")

    for painting in PAINTINGS:
        mask   = df["Painting"] == painting
        folder = os.path.join(OUTPUT_DIR, painting)
        print(f"  [{painting}]")

        for col, slug, ftype, xlabel in FEATURES:
            save_path = os.path.join(folder, f"{slug}.png")
            title     = f"{xlabel}\n— {painting} —"

            if ftype in ("numeric", "likert", "price"):
                # Data is already clean — read directly as float array
                vals = df.loc[mask, col].values.astype(float)

                if ftype == "numeric":
                    plot_numeric_histogram(painting, vals, xlabel, title, save_path)
                elif ftype == "likert":
                    plot_likert_histogram(painting, vals, xlabel, title, save_path)
                elif ftype == "price":
                    plot_price_histogram(painting, vals, title, save_path)

            elif ftype == "multi_category":
                labels = expand_multi_label(df.loc[mask, col])
                plot_category_histogram(painting, labels, xlabel, title, save_path)

            print(f"    ✓ {slug}.png")

        print()

    print(f"Done. All histograms saved to '{OUTPUT_DIR}/'")
    for painting in PAINTINGS:
        print(f"    {painting}/  ({len(FEATURES)} plots)")


if __name__ == "__main__":
    main()
