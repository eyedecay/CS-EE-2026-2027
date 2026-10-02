import csv
import glob
import json
import os
import re
import statistics

K_VALUES = [1, 2, 3]
DATASET = "financebench"
METHODS = ["sparse", "dense"]
RESULTS_DIR = "results/financebench"
TRIALS_SUBDIR = os.path.join(RESULTS_DIR, "trials")
QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]

ACCURACY_METRICS = [f"{m}@{k}" for k in K_VALUES for m in ("recall", "precision", "mrr")]
TIMING_METRICS = {
    "sparse": [
        "query_tokenize_time_ms",
        "score_time_ms",
        "retrieval_time_ms",
        "score_to_first_token_ms",
        "prompt_prep_time_ms",
        "full_ttft_ms",
        "full_gen_time_ms",
    ],
    "dense": [
        "query_embed_time_ms",
        "search_time_ms",
        "retrieval_time_ms",
        "search_to_first_token_ms",
        "prompt_prep_time_ms",
        "full_ttft_ms",
        "full_gen_time_ms",
    ],
}
REPORT_METRICS = {
    "sparse": [
        ("Query tokenize time", "query_tokenize_time_ms"),
        ("Score time", "score_time_ms"),
        ("Score to first token time", "score_to_first_token_ms"),
        ("Full TTFT", "full_ttft_ms"),
        ("Full generation time", "full_gen_time_ms"),
    ],
    "dense": [
        ("Query embedding time", "query_embed_time_ms"),
        ("Search time", "search_time_ms"),
        ("Search to first token time", "search_to_first_token_ms"),
        ("Full TTFT", "full_ttft_ms"),
        ("Full generation time", "full_gen_time_ms"),
    ],
}


def load_trials(method):
    """
    Load every recorded trial for one method, keyed by trial label
    Args:
        method (str): "sparse" or "dense"
    Returns:
        dict: Mapping from trial label (e.g. "trial_01") to loaded result list
    """
    pattern = os.path.join(TRIALS_SUBDIR, method, f"{method}_results_trial_*.json")
    trials = {}
    for path in sorted(glob.glob(pattern)):
        label = re.search(r"(trial_\d+)", os.path.basename(path)).group(1)
        with open(path) as f:
            trials[label] = json.load(f)
    return trials


def mean_std(values):
    """
    Compute mean and sample standard deviation of a list of values
    Args:
        values (list): Numeric values
    Returns:
        tuple: (mean, std) rounded to 4 decimals; both None when values is empty
    """
    if not values:
        return None, None
    std = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
    return round(statistics.mean(values), 4), std


def per_trial_stats(results, metric_names, label_filter=None):
    """
    Compute mean metrics for a single trial, optionally restricted to a question type
    Args:
        results (list): Per-query result dicts from one trial
        metric_names (list): Metric keys to aggregate
        label_filter (str or None): Question type to keep, or None for all queries
    Returns:
        dict: Mapping from metric key to mean value, or None when unavailable
    """
    subset = results if label_filter is None else [r for r in results if r["question_type"] == label_filter]
    stats = {}
    for metric in metric_names:
        values = [r[metric] for r in subset if r.get(metric) is not None]
        stats[metric] = round(statistics.mean(values), 4) if values else None
    return stats


def aggregate_over_trials(trials, metric_names):
    """
    Aggregate metrics across trials using mean of trial means and std across trial means
    Args:
        trials (dict): Mapping from trial label to loaded result list
        metric_names (list): Metric keys to aggregate
    Returns:
        dict: Mapping from metric key to mean, std across trials, and the trial means
    """
    trial_means = {m: [] for m in metric_names}
    for results in trials.values():
        stats = per_trial_stats(results, metric_names)
        for metric in metric_names:
            if stats[metric] is not None:
                trial_means[metric].append(stats[metric])

    agg = {"n_trials": len(trials)}
    for metric in metric_names:
        mean, std = mean_std(trial_means[metric])
        agg[metric] = {
            "mean": mean,
            "std_across_trials": std,
            "n_trials_used": len(trial_means[metric]),
            "trial_means": [round(v, 4) for v in trial_means[metric]],
        }
    return agg


def check_determinism(trials, method):
    """
    Verify accuracy metrics are identical across every trial of a method
    Args:
        trials (dict): Mapping from trial label to loaded result list
        method (str): "sparse" or "dense", used for reporting
    Returns:
        bool: True when all trials agree on every accuracy metric
    """
    if not trials:
        print(f"  WARNING: {method} has no trials loaded")
        return False

    fields = ACCURACY_METRICS + ["query_id", "question_type"]
    reference = None
    for label, results in trials.items():
        signature = [tuple(r[f] for f in fields) for r in results]
        if reference is None:
            reference = signature
        elif signature != reference:
            print(f"  WARNING: {method} accuracy differs in trial {label}")
            return False
    return True


def save_per_trial_csv(trial_tables, path):
    """
    Write one row per trial per metric holding that trial's mean
    Args:
        trial_tables (dict): Nested mapping method -> trial label -> question type -> metric means
        path (str): Output CSV path
    """
    fieldnames = ["method", "trial", "question_type", "metric", "value"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            for label, by_type in trial_tables[method].items():
                for qtype, stats in by_type.items():
                    qname = "all" if qtype is None else qtype
                    for metric, value in stats.items():
                        if value is None:
                            continue
                        writer.writerow({
                            "method": method,
                            "trial": label,
                            "question_type": qname,
                            "metric": metric,
                            "value": value,
                        })


def save_aggregate_csv(aggregates, path):
    """
    Write the cross-trial aggregate table with mean and std across trials
    Args:
        aggregates (dict): Nested mapping method -> question type -> metric -> aggregate stats
        path (str): Output CSV path
    """
    rows = []
    for method in METHODS:
        for qtype, metrics in aggregates[method].items():
            for metric, stats in metrics.items():
                if not isinstance(stats, dict) or stats["mean"] is None:
                    continue
                rows.append({
                    "method": method,
                    "question_type": qtype,
                    "metric": metric,
                    "mean_of_trial_means": stats["mean"],
                    "std_across_trials": stats["std_across_trials"],
                    "n_trials": stats["n_trials_used"],
                })

    fieldnames = ["method", "question_type", "metric", "mean_of_trial_means", "std_across_trials", "n_trials"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_latency_summary_csv(aggregates, path):
    """
    Write the reported latency rows per method as mean and std across trials
    Args:
        aggregates (dict): Nested mapping method -> question type -> metric -> aggregate stats
        path (str): Output CSV path
    """
    fieldnames = ["method", "metric_label", "metric", "mean_ms", "std_ms", "n_trials"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            for label, metric in REPORT_METRICS[method]:
                stats = aggregates[method]["all"][metric]
                if stats["mean"] is None:
                    continue
                writer.writerow({
                    "method": method,
                    "metric_label": label,
                    "metric": metric,
                    "mean_ms": stats["mean"],
                    "std_ms": stats["std_across_trials"],
                    "n_trials": stats["n_trials_used"],
                })


def print_latency_report(aggregates):
    """
    Print the reported latency rows per method as means across trials
    Args:
        aggregates (dict): Nested mapping method -> question type -> metric -> aggregate stats
    """
    n_trials = aggregates["dense"]["all"]["n_trials"]
    print(f"\nLatency across {n_trials} trials (mean of trial means), ms:")
    for method in METHODS:
        print(f"\n  {method.capitalize()}")
        print(f"    {'metric':28}{'mean':>10}")
        for label, metric in REPORT_METRICS[method]:
            stats = aggregates[method]["all"][metric]
            if stats["mean"] is None:
                continue
            print(f"    {label:28}{stats['mean']:10.2f}")


def main():
    """
    Aggregate every recorded trial per method and write the per-trial and
    cross-trial metric tables
    """
    os.makedirs(TRIALS_SUBDIR, exist_ok=True)

    trials_by_method = {m: load_trials(m) for m in METHODS}
    for method in METHODS:
        count = len(trials_by_method[method])
        print(f"{method}: {count} trials loaded")
        if count == 0:
            print(f"  WARNING: no {method}_results_trial_*.json files found in "
              f"{os.path.join(TRIALS_SUBDIR, method)}; run run_trials.py first")

    print("\nDeterminism check on accuracy metrics across trials:")
    deterministic = {}
    for method in METHODS:
        deterministic[method] = check_determinism(trials_by_method[method], method)
        print(f"  {method}: {'identical across all trials' if deterministic[method] else 'DIFFERS'}")

    trial_tables = {}
    for method in METHODS:
        metric_names = ACCURACY_METRICS + TIMING_METRICS[method]
        trial_tables[method] = {}
        for label, results in trials_by_method[method].items():
            by_type = {None: per_trial_stats(results, metric_names)}
            for qtype in QUESTION_TYPES:
                by_type[qtype] = per_trial_stats(results, metric_names, label_filter=qtype)
            trial_tables[method][label] = by_type

    per_trial_path = os.path.join(TRIALS_SUBDIR, "per_trial_metrics.csv")
    save_per_trial_csv(trial_tables, per_trial_path)
    print(f"\nSaved {per_trial_path}")

    aggregates = {}
    for method in METHODS:
        metric_names = ACCURACY_METRICS + TIMING_METRICS[method]
        trials = trials_by_method[method]
        aggregates[method] = {"all": aggregate_over_trials(trials, metric_names)}
        for qtype in QUESTION_TYPES:
            scoped = {label: [r for r in res if r["question_type"] == qtype] for label, res in trials.items()}
            aggregates[method][qtype] = aggregate_over_trials(scoped, metric_names)

    agg_path = os.path.join(TRIALS_SUBDIR, "aggregate_metrics.csv")
    save_aggregate_csv(aggregates, agg_path)
    print(f"Saved {agg_path}")

    latency_path = os.path.join(TRIALS_SUBDIR, "latency_summary.csv")
    save_latency_summary_csv(aggregates, latency_path)
    print(f"Saved {latency_path}")

    json_path = os.path.join(TRIALS_SUBDIR, "aggregate_metrics.json")
    payload = {
        "dataset": DATASET,
        "n_trials": {m: len(trials_by_method[m]) for m in METHODS},
        "accuracy_identical_across_trials": deterministic,
        "aggregate": aggregates,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {json_path}")

    if any(trials_by_method[m] for m in METHODS):
        print_latency_report(aggregates)
        print("\n  Standard deviations are recorded in "
              f"{os.path.join(TRIALS_SUBDIR, 'latency_summary.csv')} and aggregate_metrics.csv")


if __name__ == "__main__":
    main()