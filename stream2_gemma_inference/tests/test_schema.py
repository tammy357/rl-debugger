import pytest

from stream2_gemma_inference.schema import extract_json, strip_reasoning, validate

GOOD = {
    "confirmed": ["no penalty at failure event (frames 12-13; no curve dip)"],
    "ruled_out": [],
    "next_to_check": {"step_range": [70000, 75000], "reason": "reward flat during failures"},
    "proposed_reward_edit": "add -1.0 penalty when object leaves the workspace",
}


def fenced(obj):
    import json
    return f"```json\n{json.dumps(obj)}\n```"


def test_strip_reasoning_removes_think_tags():
    assert strip_reasoning("<think>hmm secret</think>answer") == "answer"


def test_strip_reasoning_removes_channel_thought_blocks():
    assert strip_reasoning("<|channel>thought hmm <channel|>answer") == "answer"


def test_strip_reasoning_passthrough_plain_prose():
    assert strip_reasoning("just prose") == "just prose"


def test_extract_takes_last_fenced_block():
    text = fenced({"echo": "of our manifest"}) + "\nanalysis...\n" + fenced(GOOD)
    assert extract_json(text) == GOOD


def test_extract_falls_back_to_last_raw_object():
    text = 'preamble {"echo": 1} middle {"confirmed": []} end'
    assert extract_json(text) == {"confirmed": []}


def test_extract_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_validate_accepts_good(run1):
    _, _, manifest = run1
    assert validate(GOOD, manifest) == []


def test_validate_accepts_nulls():
    ok = {"confirmed": [], "ruled_out": [], "next_to_check": None, "proposed_reward_edit": None}
    assert validate(ok, None) == []


def test_validate_rejects_missing_key():
    bad = {k: v for k, v in GOOD.items() if k != "ruled_out"}
    assert any("ruled_out" in p for p in validate(bad, None))


def test_validate_rejects_descending_step_range(run1):
    _, _, manifest = run1
    bad = dict(GOOD, next_to_check={"step_range": [75000, 70000], "reason": "x"})
    assert validate(bad, manifest) != []


def test_validate_rejects_implausible_training_steps(run1):
    _, _, manifest = run1  # checkpoint_step=80000, window width 5000 -> cap 85000
    bad = dict(GOOD, next_to_check={"step_range": [90000, 95000], "reason": "x"})
    assert validate(bad, manifest) != []


def test_validate_allows_earlier_training_window(run1):
    _, _, manifest = run1  # README's own example points outside the current window
    ok = dict(GOOD, next_to_check={"step_range": [40000, 45000], "reason": "x"})
    assert validate(ok, manifest) == []


def test_validate_rejects_non_string_claims():
    bad = dict(GOOD, confirmed=[{"claim": "objectified"}])
    assert validate(bad, None) != []
