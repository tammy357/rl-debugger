import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUTPUTS = REPO_ROOT / "stream1_simulation" / "outputs"


def load_run(n):
    """Return (frames, chart, manifest) exactly as check_gemma_contract.py builds them."""
    run_dir = OUTPUTS / f"run{n}"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    frames = [str(run_dir / rec["file"]) for rec in manifest["frames"]]
    chart = str(run_dir / manifest["reward_curve"])
    return frames, chart, manifest


def bootstrap_log(n):
    """Run-1 empty log, the exact shape check_gemma_contract.py passes."""
    return {
        "run": n,
        "confirmed": [],
        "ruled_out": [],
        "next_to_check": None,
        "proposed_reward_edit": None,
    }


@pytest.fixture
def run1():
    return load_run(1)
