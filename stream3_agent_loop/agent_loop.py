"""
Stream 3 — Agent loop.

for each run:
    load hypothesis_log from Antigravity (local JSON fallback: state.py)
    call Gemma inference (Stream 2)
    update hypothesis_log
    save back to Antigravity
    call Computer Use trigger (Stream 4)
    surface result to UI (Stream 4 reads state/ directly, see app.py)

Run standalone:
    python agent_loop.py --run 1
    python agent_loop.py --run 1 --run 2 --run 3   (repeat --run for each)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stream3_agent_loop.state import load_hypothesis_log, new_hypothesis_log, save_hypothesis_log
from stream2_gemma_inference.analyze_run_stub import analyze_run  # swap for real Stream 2 module later
from stream4_demo_ui.computer_use import TimestepRequest, fetch_wandb_screenshot

STREAM1_OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "stream1_simulation", "outputs")


def load_run_inputs(run: int):
    """Loads frames + reward curve path for a run from Stream 1's manifest."""
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


def process_run(run: int) -> dict:
    """Runs one full iteration of the loop for a single run. Returns the
    updated hypothesis log."""
    frames, chart, manifest = load_run_inputs(run)

    hypothesis_log = load_hypothesis_log(run) or new_hypothesis_log(run)
    print(f"[run {run}] loaded hypothesis log: "
          f"{len(hypothesis_log['confirmed'])} confirmed, {len(hypothesis_log['ruled_out'])} ruled out")

    # manifest carries drop_step, step_range, total_reward, num_episode_steps,
    # success, etc. -- the same log data a human researcher would read
    # alongside the rollout video, not just the pixels.
    updated_log = analyze_run(frames, chart, manifest, hypothesis_log)
    save_hypothesis_log(run, updated_log)
    print(f"[run {run}] saved updated hypothesis log")

    next_check = updated_log.get("next_to_check")
    if next_check:
        req = TimestepRequest(run=next_check["run"], step_range=tuple(next_check["step_range"]))
        screenshot_path = fetch_wandb_screenshot(req)
        print(f"[run {run}] Computer Use screenshot: {screenshot_path}")

    return updated_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, action="append", required=True,
                         help="Run number to process. Repeat --run for multiple runs.")
    args = parser.parse_args()

    for run in args.run:
        try:
            process_run(run)
        except FileNotFoundError as e:
            print(f"[run {run}] SKIPPED: {e}")


if __name__ == "__main__":
    main()
