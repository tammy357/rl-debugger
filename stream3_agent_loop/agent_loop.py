"""
Stream 3 — Agent loop. Statement Five (local/offline) -- Computer Use/WandB
removed, local JSON state is the preferred path (not a fallback).

for each run:
    load hypothesis_log from local JSON state
    call Gemma inference (Stream 2's real analyze_run)
    save the returned state VERBATIM (it's complete cumulative state, not a
        delta -- do not merge, or run-1 findings vanish on run 2)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stream3_agent_loop.state import load_hypothesis_log, new_hypothesis_log, save_hypothesis_log
from stream2_gemma_inference import analyze_run, AnalyzeRunError

STREAM1_OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "stream1_simulation", "outputs")


def load_run_inputs(run: int):
    manifest_path = os.path.join(STREAM1_OUTPUTS, f"run{run}", "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No manifest for run {run} at {manifest_path} — has Stream 1 produced this run yet?"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    out_dir = os.path.join(STREAM1_OUTPUTS, f"run{run}")
    frames = [os.path.join(out_dir, record["file"]) for record in manifest["frames"]]
    chart = os.path.join(out_dir, manifest["reward_curve"])
    return frames, chart, manifest


def process_run(run: int, prior_log: dict = None) -> dict:
    frames, chart, manifest = load_run_inputs(run)

    # Carry forward the log from the previous run in THIS invocation's
    # sequence if given; otherwise fall back to this run's own last-saved
    # state (resuming a partial sequence), or a fresh log if neither exists.
    hypothesis_log = prior_log if prior_log is not None else (load_hypothesis_log(run) or new_hypothesis_log(run))
    print(f"[run {run}] loaded: {len(hypothesis_log['confirmed'])} confirmed, "
          f"{len(hypothesis_log['ruled_out'])} ruled out")

    # Real signature: analyze_run(frames, chart, hypothesis_log, manifest).
    # Always use keywords -- positional args swap silently with no error.
    try:
        updated_log = analyze_run(frames, chart, hypothesis_log=hypothesis_log, manifest=manifest)
    except AnalyzeRunError as e:
        print(f"[run {run}] analyze_run failed ({e.kind}): {e}")
        return hypothesis_log

    save_hypothesis_log(run, updated_log)
    print(f"[run {run}] saved updated hypothesis log")

    if updated_log.get("next_to_check") is None:
        print(f"[run {run}] no next_to_check flagged (converged/confident)")

    return updated_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, action="append", required=True)
    args = parser.parse_args()

    prior_log = None
    for run in args.run:
        try:
            prior_log = process_run(run, prior_log)
        except FileNotFoundError as e:
            print(f"[run {run}] SKIPPED: {e}")


if __name__ == "__main__":
    main()