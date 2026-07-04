"""Mock consumer standing in for Stream 2's analyze_run(frames, chart,
hypothesis_log) until it exists. Loads a run's manifest + frames + chart and
asserts the shape matches what that function will expect, so a format
mismatch is caught here instead of at integration time.
"""

import argparse
import json
import os

from PIL import Image

STREAM1_DIR = os.path.dirname(__file__)
EXPECTED_SIZE = (320, 320)


def check_run(run_id):
    out_dir = os.path.join(STREAM1_DIR, "outputs", f"run{run_id}")
    manifest_path = os.path.join(out_dir, "manifest.json")
    assert os.path.exists(manifest_path), f"missing {manifest_path}"

    with open(manifest_path) as f:
        manifest = json.load(f)

    for key in ("run", "checkpoint_step", "step_range", "drop_step", "frames", "reward_curve"):
        assert key in manifest, f"manifest missing required key: {key}"

    assert manifest["frames"], "manifest has no frames"

    # This is the exact shape Stream 2's analyze_run(frames, chart, hypothesis_log) will receive.
    frames = []
    for record in manifest["frames"]:
        frame_path = os.path.join(out_dir, record["file"])
        assert os.path.exists(frame_path), f"missing frame file {frame_path}"
        img = Image.open(frame_path)
        assert img.size == EXPECTED_SIZE, f"{frame_path} has size {img.size}, expected {EXPECTED_SIZE}"
        assert img.mode == "RGB", f"{frame_path} has mode {img.mode}, expected RGB"
        frames.append(frame_path)

    chart_path = os.path.join(out_dir, manifest["reward_curve"])
    assert os.path.exists(chart_path), f"missing chart {chart_path}"
    Image.open(chart_path).verify()

    # Matches README.md's hypothesis log entry shape -- a dummy stand-in for
    # what Stream 3 would actually load/pass in.
    hypothesis_log = {
        "run": manifest["run"],
        "confirmed": [],
        "ruled_out": [],
        "next_to_check": None,
        "proposed_reward_edit": None,
    }

    print(f"run{run_id}: OK -- {len(frames)} frames @ {EXPECTED_SIZE}, chart present, "
          f"drop_step={manifest['drop_step']}, step_range={manifest['step_range']}")
    print(f"  analyze_run(frames=<{len(frames)} paths>, chart='{chart_path}', "
          f"hypothesis_log={hypothesis_log})")
    return frames, chart_path, hypothesis_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    args = parser.parse_args()
    check_run(args.run_id)


if __name__ == "__main__":
    main()
