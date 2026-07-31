import csv
import json
import os
import statistics
from collections import defaultdict

K_VALUES = [1, 2, 3]
DATASET = "financebench"
RESULTS_DIR = "results/financebench"


def load(path):
    """
    Load JSON results from a file
    Args:
        path (str): Path to the JSON file
    Returns:
        list: Parsed JSON result dicts
    """
    with open(path) as f:
        return json.load(f)


def mean_std(vals):
    """
    Compute mean and standard deviation of a list of values
    Args:
        vals (list): Numeric values
    Returns:
        tuple: (mean, std) rounded to 4 decimal places; std is 0.0 for single values
    """
    return round(statistics.mean(vals), 4), round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0


def aggregate(results, label, has_embed):
    """
    Aggregate per-query metrics into overall summary statistics
    Args:
        results (list): List of per-query result dicts
        label (str): Method label for the summary
        has_embed (bool): Whether results include embedding times (dense) or tokenize/index times (sparse)
    Returns:
        dict: Aggregated metrics with mean and std for each metric
    """
    n = len(results)
    agg = {"method": label, "n_queries": n, "dataset": DATASET}

    for k in K_VALUES:
        for metric in ["recall", "precision", "mrr"]:
            key = f"{metric}@{k}"
            agg[key] = {}
            agg[key]["mean"], agg[key]["std"] = mean_std([r[key] for r in results])

    agg["retrieval_time_ms"] = {}
    agg["retrieval_time_ms"]["mean"], agg["retrieval_time_ms"]["std"] = mean_std([r["retrieval_time_ms"] for r in results])

    if has_embed:
        agg["query_embed_time_ms"] = {}
        agg["query_embed_time_ms"]["mean"], agg["query_embed_time_ms"]["std"] = mean_std([r["query_embed_time_ms"] for r in results])
        agg["search_time_ms"] = {}
        agg["search_time_ms"]["mean"], agg["search_time_ms"]["std"] = mean_std([r["search_time_ms"] for r in results])
    else:
        agg["tokenize_time_ms"] = {}
        agg["tokenize_time_ms"]["mean"], agg["tokenize_time_ms"]["std"] = mean_std([r["tokenize_time_ms"] for r in results])
        agg["score_time_ms"] = {}
        agg["score_time_ms"]["mean"], agg["score_time_ms"]["std"] = mean_std([r["score_time_ms"] for r in results])

    ttft_vals = [r["ttft_ms"] for r in results if r.get("ttft_ms") is not None]
    if ttft_vals:
        agg["ttft_ms"] = {}
        agg["ttft_ms"]["mean"], agg["ttft_ms"]["std"] = mean_std(ttft_vals)
        agg["ttft_ms"]["n"] = len(ttft_vals)
        gen_vals = [r["gen_time_ms"] for r in results if r.get("gen_time_ms") is not None]
        agg["gen_time_ms"] = {}
        agg["gen_time_ms"]["mean"], agg["gen_time_ms"]["std"] = mean_std(gen_vals)
        agg["gen_time_ms"]["n"] = len(gen_vals)

    return agg


def aggregate_by_type(results, label, has_embed):
    """
    Aggregate results broken down by question_type
    Args:
        results (list): List of per-query result dicts
        label (str): Method label prefix
        has_embed (bool): Whether results include embedding times
    Returns:
        dict: Mapping from question_type to aggregated summary dict
    """
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)
    return {qt: aggregate(qr, f"{label} ({qt})", has_embed) for qt, qr in by_type.items()}


def compute_comparison(sparse_agg, dense_agg):
    """
    Compare sparse and dense aggregated results, computing deltas
    Args:
        sparse_agg (dict): Aggregated sparse retrieval metrics
        dense_agg (dict): Aggregated dense retrieval metrics
    Returns:
        dict: Comparison table with sparse, dense, delta, and better for each metric
    """
    comp = {}
    for k in K_VALUES:
        for metric in ["recall", "precision", "mrr"]:
            key = f"{metric}@{k}"
            s = sparse_agg[key]["mean"]
            d = dense_agg[key]["mean"]
            comp[key] = {"sparse": s, "dense": d, "delta": round(d - s, 4), "better": "dense" if d > s else "sparse"}

    comp["retrieval_time_ms"] = {
        "sparse": sparse_agg["retrieval_time_ms"]["mean"],
        "dense": dense_agg["retrieval_time_ms"]["mean"],
        "delta": round(dense_agg["retrieval_time_ms"]["mean"] - sparse_agg["retrieval_time_ms"]["mean"], 2),
        "better": "sparse" if sparse_agg["retrieval_time_ms"]["mean"] < dense_agg["retrieval_time_ms"]["mean"] else "dense",
    }

    if "ttft_ms" in sparse_agg and "ttft_ms" in dense_agg:
        comp["ttft_ms"] = {
            "sparse": sparse_agg["ttft_ms"]["mean"],
            "dense": dense_agg["ttft_ms"]["mean"],
            "delta": round(dense_agg["ttft_ms"]["mean"] - sparse_agg["ttft_ms"]["mean"], 2),
            "better": "sparse" if sparse_agg["ttft_ms"]["mean"] < dense_agg["ttft_ms"]["mean"] else "dense",
        }

    return comp


def save_csv(agg_sparse, agg_dense, comp, path):
    """
    Write comparison metrics to a CSV file
    Args:
        agg_sparse (dict): Aggregated sparse metrics
        agg_dense (dict): Aggregated dense metrics
        comp (dict): Comparison between the two
        path (str): Output CSV path
    """
    rows = []
    for k in K_VALUES:
        for metric in ["recall", "precision", "mrr"]:
            key = f"{metric}@{k}"
            rows.append({
                "metric": key,
                "sparse_mean": agg_sparse[key]["mean"],
                "sparse_std": agg_sparse[key]["std"],
                "dense_mean": agg_dense[key]["mean"],
                "dense_std": agg_dense[key]["std"],
                "delta": comp[key]["delta"],
                "better": comp[key]["better"],
            })

    for metric in ["retrieval_time_ms", "ttft_ms"]:
        if metric in comp:
            rows.append({
                "metric": metric,
                "sparse_mean": agg_sparse[metric]["mean"] if metric in agg_sparse else "",
                "sparse_std": agg_sparse[metric]["std"] if metric in agg_sparse else "",
                "dense_mean": agg_dense[metric]["mean"] if metric in agg_dense else "",
                "dense_std": agg_dense[metric]["std"] if metric in agg_dense else "",
                "delta": comp[metric]["delta"],
                "better": comp[metric]["better"],
            })

    fieldnames = ["metric", "sparse_mean", "sparse_std", "dense_mean", "dense_std", "delta", "better"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_csv_by_type(sparse_by_type, dense_by_type, path):
    """
    Write per-question-type comparison metrics to a CSV file
    Args:
        sparse_by_type (dict): Sparse results aggregated by question_type
        dense_by_type (dict): Dense results aggregated by question_type
        path (str): Output CSV path
    """
    rows = []
    for qt in sorted(sparse_by_type):
        s = sparse_by_type[qt]
        d = dense_by_type.get(qt, {})
        comp = compute_comparison(s, d)
        for k in K_VALUES:
            for metric in ["recall", "precision", "mrr"]:
                key = f"{metric}@{k}"
                rows.append({
                    "question_type": qt,
                    "metric": key,
                    "sparse_mean": s[key]["mean"],
                    "sparse_std": s[key]["std"],
                    "dense_mean": d[key]["mean"],
                    "dense_std": d[key]["std"],
                    "delta": comp[key]["delta"],
                    "better": comp[key]["better"],
                })

    fieldnames = ["question_type", "metric", "sparse_mean", "sparse_std", "dense_mean", "dense_std", "delta", "better"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sparse_path = os.path.join(RESULTS_DIR, "sparse_results.json")
    dense_path = os.path.join(RESULTS_DIR, "dense_results.json")

    sparse_raw = load(sparse_path)
    dense_raw = load(dense_path)

    agg_sparse = aggregate(sparse_raw, "SPARSE — BM25 (bm25s)", has_embed=False)
    agg_dense = aggregate(dense_raw, "DENSE — FAISS + all-MiniLM-L6-v2", has_embed=True)
    comp = compute_comparison(agg_sparse, agg_dense)

    summary = {"sparse": agg_sparse, "dense": agg_dense, "comparison": comp}
    json_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    save_csv(agg_sparse, agg_dense, comp, csv_path)

    types_sparse = aggregate_by_type(sparse_raw, "SPARSE — BM25 (bm25s)", has_embed=False)
    types_dense = aggregate_by_type(dense_raw, "DENSE — FAISS + all-MiniLM-L6-v2", has_embed=True)
    types_path = os.path.join(RESULTS_DIR, "summary_by_type.csv")
    save_csv_by_type(types_sparse, types_dense, types_path)

    print(f"Saved {json_path}, {csv_path}, and {types_path}")
