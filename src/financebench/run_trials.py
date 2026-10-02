import glob
import os
import subprocess
import sys
import time

METHODS = ["sparse", "dense"]
DEFAULT_TRIALS = 10
DEFAULT_DEVICE = "cpu"
RESULTS_DIR = "results/financebench"
TRIALS_SUBDIR = os.path.join(RESULTS_DIR, "trials")
LOG_PATH = os.path.join(TRIALS_SUBDIR, "trials_run.log")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def clear_stale_trials():
    """
    Delete trial outputs from any previous run so that aggregates cannot mix
    trials from different configurations
    Returns:
        list: Paths that were removed
    """
    removed = []
    for method in METHODS:
        method_dir = os.path.join(RESULTS_DIR, "trials", method)
        patterns = [
            os.path.join(method_dir, f"{method}_results_trial_*.json"),
            os.path.join(method_dir, f"{method}_trial_*_avg.json"),
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                os.remove(path)
                removed.append(path)
    return removed


def run_pass(script, tag, warmup, step, total):
    """
    Run one experiment pass in a subprocess, appending its output to the run log
    Args:
        script (str): Absolute path to the experiment script
        tag (str): Output filename tag, e.g. "_trial_01" or "_warmup"
        warmup (bool): Whether this is the discarded warmup pass
        step (int): 1-based position of this pass in the overall run
        total (int): Total number of passes in the run
    Raises:
        SystemExit: If the pass exits with a non-zero return code
    """
    label = "warmup" if warmup else tag.lstrip("_")
    env = dict(os.environ)
    env["FINBENCH_TAG"] = tag
    env["FINBENCH_WARMUP"] = "1" if warmup else "0"
    env.setdefault("FINBENCH_DEVICE", DEFAULT_DEVICE)

    print(f"[{step}/{total}] {os.path.basename(script)} {label} ... ", end="", flush=True)
    started = time.perf_counter()
    with open(LOG_PATH, "a") as log:
        log.write(f"\n=== {os.path.basename(script)} {label} ===\n")
        log.flush()
        result = subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        print(f"FAILED after {elapsed:.1f}s (exit {result.returncode})")
        raise SystemExit(f"{os.path.basename(script)} {label} failed; see {LOG_PATH}")

    print(f"done in {elapsed:.1f}s")


def main():
    """
    Run the warmup and recorded trials for both methods, then analyze the results
    """
    os.chdir(ROOT)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    trials = int(os.environ.get("FINBENCH_TRIALS", DEFAULT_TRIALS))
    device = os.environ.get("FINBENCH_DEVICE", DEFAULT_DEVICE)
    total = trials * len(METHODS) + len(METHODS)

    stale = clear_stale_trials()
    if stale:
        print(f"Removed {len(stale)} trial file(s) from a previous run:")
        for path in stale:
            print(f"  {path}")

    print(f"Device: {device} | Recorded trials per method: {trials}\n")
    open(LOG_PATH, "w").close()

    step = 0
    for method in METHODS:
        script = os.path.join(HERE, f"{method}-experiment.py")
        step += 1
        run_pass(script, "_warmup", True, step, total)

        for i in range(1, trials + 1):
            step += 1
            run_pass(script, f"_trial_{i:02d}", False, step, total)

    print("\nAll trials complete, running analyze.py ...\n")
    subprocess.run([sys.executable, os.path.join(HERE, "analyze.py")], cwd=ROOT, check=True)

    print("\nRunning plots_trials.py ...\n")
    subprocess.run([sys.executable, os.path.join(HERE, "plots_trials.py")], cwd=ROOT, check=True)
    print("\nDone. Reported metrics are in "
          f"{os.path.join(RESULTS_DIR, 'trials', 'latency_summary.csv')}")


if __name__ == "__main__":
    main()