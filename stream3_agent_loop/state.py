"""
Stream 3 — Local JSON persistence for hypothesis logs (this is the
preferred path for Statement Five's offline/privacy-first story, not a
fallback).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
os.makedirs(STATE_DIR, exist_ok=True)


def _path(run: int) -> str:
    return os.path.join(STATE_DIR, f"run_{run}_hypothesis.json")


def load_hypothesis_log(run: int) -> dict | None:
    path = _path(run)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_hypothesis_log(run: int, data: dict) -> None:
    """Overwrites with the complete cumulative state -- callers must pass
    the full state, not a delta (see Stream 2's contract notes)."""
    with open(_path(run), "w") as f:
        json.dump(data, f, indent=2)


def new_hypothesis_log(run: int) -> dict:
    return {
        "run": run,
        "confirmed": [],
        "ruled_out": [],
        "next_to_check": None,
        "proposed_reward_edit": None,
    }