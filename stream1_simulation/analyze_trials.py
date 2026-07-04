"""Aggregate run_trials.py's per-episode trial log into per-checkpoint
bug-reproduction-rate statistics: outputs/trial_summary.json and
outputs/bug_rate_curve.png, plus a console summary suitable for demo
narration.

Usage:
    python analyze_trials.py
    python analyze_trials.py --trial_log outputs/trial_log.jsonl
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from record_utils import plot_bug_rate_curve

STREAM1_DIR = os.path.dirname(__file__)
DEFAULT_TRIAL_LOG = os.path.join(STREAM1_DIR, "outputs", "trial_log.jsonl")
DEFAULT_SUMMARY_PATH = os.path.join(STREAM1_DIR, "outputs", "trial_summary.json")
DEFAULT_CHART_PATH = os.path.join(STREAM1_DIR, "outputs", "bug_rate_curve.png")


def load_records(trial_log_path):
    records = []
    with open(trial_log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate(records):
    by_checkpoint = defaultdict(list)
    seeds = set()
    for r in records:
        by_checkpoint[r["checkpoint_step"]].append(r)
        seeds.add(r["seed"])

    per_checkpoint = []
    for step in sorted(by_checkpoint):
        rows = by_checkpoint[step]
        n = len(rows)
        n_dropped = sum(1 for r in rows if r["dropped"])
        n_success = sum(1 for r in rows if r["success"])
        rewards = np.array([r["total_reward"] for r in rows], dtype=float)
        per_checkpoint.append({
            "checkpoint_step": step,
            "n_episodes": n,
            "n_dropped": n_dropped,
            "n_success": n_success,
            "bug_rate": n_dropped / n,
            "success_rate": n_success / n,
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
        })

    return {
        "n_seeds": len(seeds),
        "n_total_episodes": len(records),
        "per_checkpoint": per_checkpoint,
    }


def print_summary(summary):
    per_checkpoint = summary["per_checkpoint"]
    if not per_checkpoint:
        print("No trial data found.")
        return

    first, last = per_checkpoint[0], per_checkpoint[-1]
    n_per_checkpoint = first["n_episodes"]
    print(
        f"Across {summary['n_seeds']} seeds ({n_per_checkpoint} rollouts/checkpoint total, "
        f"{summary['n_total_episodes']} episodes overall): bug reproduction rate is "
        f"{first['bug_rate']:.0%} at step {first['checkpoint_step']}, "
        f"changing to {last['bug_rate']:.0%} at step {last['checkpoint_step']}."
    )
    print(f"{'step':>10}  {'n':>5}  {'bug_rate':>9}  {'success_rate':>13}  {'mean_reward':>12}")
    for row in per_checkpoint:
        print(
            f"{row['checkpoint_step']:>10}  {row['n_episodes']:>5}  "
            f"{row['bug_rate']:>9.0%}  {row['success_rate']:>13.0%}  {row['mean_reward']:>12.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_log", default=DEFAULT_TRIAL_LOG)
    parser.add_argument("--summary_out", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--chart_out", default=DEFAULT_CHART_PATH)
    args = parser.parse_args()

    records = load_records(args.trial_log)
    if not records:
        raise SystemExit(f"No records found in {args.trial_log} -- run run_trials.py first.")

    summary = aggregate(records)

    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    steps = [row["checkpoint_step"] for row in summary["per_checkpoint"]]
    bug_rates = [row["bug_rate"] for row in summary["per_checkpoint"]]
    plot_bug_rate_curve(
        steps, bug_rates,
        n_seeds=summary["n_seeds"],
        n_episodes_per_checkpoint=summary["per_checkpoint"][0]["n_episodes"],
        out_path=args.chart_out,
    )  # n_episodes_per_checkpoint here is the total across all seeds at that step

    print_summary(summary)
    print(f"\nWrote {args.summary_out} and {args.chart_out}")


if __name__ == "__main__":
    main()
