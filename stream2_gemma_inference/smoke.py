"""Manual live smoke test — requires a running Gemma server. NOT run in CI.

Hour-0 vision gate: send two DIFFERENT images in one request; the model must
describe both. Catches silently-text-only vision setups (wrong/missing
projector) before anything is built on top. Then one timed analyze_run.

Usage:
    python -m stream2_gemma_inference.smoke --run_id 1 [--gate-only]
"""

import argparse
import json
import time
from pathlib import Path

from .analyze import analyze_run
from .client import GemmaClient
from .prompts import _image, _text

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "stream1_simulation" / "outputs"


def load_run(n):
    run_dir = OUTPUTS / f"run{n}"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    frames = [str(run_dir / rec["file"]) for rec in manifest["frames"]]
    chart = str(run_dir / manifest["reward_curve"])
    return frames, chart, manifest


def vision_gate(client, frames, chart):
    print("== HOUR-0 VISION GATE: two images, model must describe both ==")
    messages = [{"role": "user", "content": [
        _text("Two images follow. Reply with exactly two lines:\n"
              "IMAGE 1: <one-sentence description>\nIMAGE 2: <one-sentence description>"),
        _image(frames[0]),
        _image(chart),
    ]}]
    reply = client.chat(messages)
    print(reply)
    ok = "IMAGE 1" in reply and "IMAGE 2" in reply
    print(f"gate {'PASSED' if ok else 'FAILED - fix vision setup before continuing'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()

    client = GemmaClient()
    frames, chart, manifest = load_run(args.run_id)
    if not vision_gate(client, frames, chart) or args.gate_only:
        return

    log = {"run": manifest["run"], "confirmed": [], "ruled_out": [],
           "next_to_check": None, "proposed_reward_edit": None}
    print(f"\n== analyze_run(run{args.run_id}) ==")
    t0 = time.monotonic()
    result = analyze_run(frames, chart, log, manifest, client=client)
    elapsed = time.monotonic() - t0
    print(json.dumps(result, indent=2))
    print(f"\nlatency: {elapsed:.1f}s (target <60s) | usage: {client.last_usage}")


if __name__ == "__main__":
    main()
