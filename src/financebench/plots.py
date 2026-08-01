import csv
import json
import os

import matplotlib.pyplot as plt

RESULTS_DIR = "results/financebench"
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")
K_VALUES = [1, 2, 3]
QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]
SPARSE_COLOR = "#6a7f8f"
DENSE_COLOR = "#1d3557"


def load_summary():
    """
    Load overall aggregate metrics from summary.json
    Returns:
        dict: Mapping from method name to metric dicts with "mean"
    """
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        return json.load(f)


def load_by_type():
    """
    Load per-question-type aggregate metrics from summary_by_type.csv
    Returns:
        list: Rows as dicts with keys from the CSV header
    """
    with open(os.path.join(RESULTS_DIR, "summary_by_type.csv")) as f:
        return list(csv.DictReader(f))


def mean_by_type(rows, qtype, metric):
    """
    Get the sparse and dense means for one metric and question type
    Args:
        rows (list): Rows from summary_by_type.csv
        qtype (str): Question type
        metric (str): Metric key, e.g. "recall@3"
    Returns:
        tuple: (sparse mean, dense mean) as floats
    """
    for row in rows:
        if row["question_type"] == qtype and row["metric"] == metric:
            return float(row["sparse_mean"]), float(row["dense_mean"])
    raise ValueError(f"Missing {metric} for {qtype}")


def annotate_bars(ax, bars, fmt):
    """
    Label bars with their values
    Args:
        ax (Axes): Matplotlib axes containing the bars
        bars (BarContainer): Bars to label
        fmt (str): Format string for the labels
    """
    ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)


def plot_accuracy_all():
    """
    Plot grouped bars of recall@k, precision@k, and mrr@k for both methods
    """
    summary = load_summary()
    metric_names = ["recall", "precision", "mrr"]
    labels = [f"@{k}" for k in K_VALUES]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    width = 0.35
    for ax, name in zip(axes, metric_names):
        sparse_vals = [summary["sparse"][f"{name}@{k}"]["mean"] for k in K_VALUES]
        dense_vals = [summary["dense"][f"{name}@{k}"]["mean"] for k in K_VALUES]
        x = range(len(labels))
        b1 = ax.bar([i - width / 2 for i in x], sparse_vals, width, color=SPARSE_COLOR, label="Sparse")
        b2 = ax.bar([i + width / 2 for i in x], dense_vals, width, color=DENSE_COLOR, label="Dense")
        annotate_bars(ax, b1, "%.4f")
        annotate_bars(ax, b2, "%.4f")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(name.capitalize(), fontsize=10)
        ax.set_ylim(0, 0.7)
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=8)
    fig.supxlabel("Top-k value")
    fig.suptitle("Retrieval accuracy (all metrics)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "accuracy_all.png"), dpi=300)
    plt.close(fig)


def plot_by_type_all():
    """
    Plot one chart per question type with recall@k, precision@k, and mrr@k panels
    """
    rows = load_by_type()
    metric_names = ["recall", "precision", "mrr"]
    labels = [f"@{k}" for k in K_VALUES]

    for qtype in QUESTION_TYPES:
        fig, axes = plt.subplots(1, len(metric_names), figsize=(12, 4), sharey=True)
        width = 0.35
        for ax, name in zip(axes, metric_names):
            sparse_vals = []
            dense_vals = []
            for k in K_VALUES:
                sparse_mean, dense_mean = mean_by_type(rows, qtype, f"{name}@{k}")
                sparse_vals.append(sparse_mean)
                dense_vals.append(dense_mean)
            x = range(len(labels))
            b1 = ax.bar([pos - width / 2 for pos in x], sparse_vals, width, color=SPARSE_COLOR, label="Sparse")
            b2 = ax.bar([pos + width / 2 for pos in x], dense_vals, width, color=DENSE_COLOR, label="Dense")
            annotate_bars(ax, b1, "%.4f")
            annotate_bars(ax, b2, "%.4f")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(name.capitalize(), fontsize=10)
            ax.set_ylim(0, 0.7)
        axes[0].set_ylabel("Value")
        axes[0].legend(fontsize=8)
        fig.supxlabel("Top-k value")
        fig.suptitle(f"{qtype} (N=50)")
        fig.tight_layout()
        filename = qtype.replace("-", "_")
        fig.savefig(os.path.join(FIGURE_DIR, f"accuracy_by_type_{filename}.png"), dpi=300)
        plt.close(fig)


def main():
    """
    Generate accuracy figures from aggregate results
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_accuracy_all()
    plot_by_type_all()
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
