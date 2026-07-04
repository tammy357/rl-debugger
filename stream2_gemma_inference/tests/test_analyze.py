import json
from datetime import datetime

import pytest

from stream2_gemma_inference import analyze_run
from stream2_gemma_inference.errors import AnalyzeRunError
from stream2_gemma_inference.mock_client import MockClient
from stream2_gemma_inference.tests.conftest import bootstrap_log, load_run


def model_json(confirmed=None, step_range=(70000, 75000)):
    return {
        "confirmed": confirmed or ["no penalty at failure event (frames 12-13; no curve dip)"],
        "ruled_out": [],
        "next_to_check": {"step_range": list(step_range), "reason": "reward flat during failures"},
        "proposed_reward_edit": "add -1.0 penalty when the object leaves the workspace",
    }


def as_response(obj):
    return "STAGE 1 ... STAGE 4:\n```json\n" + json.dumps(obj) + "\n```"


def test_happy_path_stamps_bookkeeping(run1):
    frames, chart, manifest = run1
    mock = MockClient([as_response(model_json())])
    result = analyze_run(frames, chart, bootstrap_log(1), manifest, client=mock)
    assert result["run"] == manifest["run"]
    assert result["next_to_check"]["run"] == manifest["run"]
    datetime.fromisoformat(result["timestamp"])  # valid ISO stamp
    assert result["confirmed"] == model_json()["confirmed"]


def test_default_eight_frames_with_drop_pinned(run1):
    frames, chart, manifest = run1
    mock = MockClient([as_response(model_json())])
    analyze_run(frames, chart, bootstrap_log(1), manifest, client=mock)
    user = mock.calls[0]["messages"][1]
    images = [p for p in user["content"] if p["type"] == "image_url"]
    texts = "\n".join(p["text"] for p in user["content"] if p["type"] == "text")
    assert len(images) == 9  # chart + 8 frames (default budget)
    assert f"sim step {manifest['drop_step']}" in texts  # drop frame survived


def test_contact_sheet_mode_sends_two_images(run1):
    frames, chart, manifest = run1
    mock = MockClient([as_response(model_json())])
    analyze_run(frames, chart, bootstrap_log(1), manifest, client=mock, contact_sheet=True)
    user = mock.calls[0]["messages"][1]
    images = [p for p in user["content"] if p["type"] == "image_url"]
    assert len(images) == 2  # chart + one contact sheet


def test_degraded_no_manifest_path(run1):
    frames, chart, _ = run1
    mock = MockClient([as_response(model_json())])  # model DOES emit next_to_check
    result = analyze_run(frames, chart, bootstrap_log(1), client=mock)
    assert result["next_to_check"] is None  # forced null - garbage can't reach Stream 4
    assert result["run"] == 1  # from hypothesis_log


def test_repair_round_is_text_only(run1):
    frames, chart, manifest = run1
    mock = MockClient(["no json at all here", as_response(model_json())])
    result = analyze_run(frames, chart, bootstrap_log(1), manifest, client=mock)
    assert result["confirmed"] == model_json()["confirmed"]
    repair = mock.calls[1]
    assert repair["temperature"] == 0
    assert all(isinstance(m["content"], str) for m in repair["messages"])  # no images


def test_two_failures_raise_bad_json(run1):
    frames, chart, manifest = run1
    mock = MockClient(["garbage", "still garbage"])
    with pytest.raises(AnalyzeRunError) as exc:
        analyze_run(frames, chart, bootstrap_log(1), manifest, client=mock)
    assert exc.value.kind == "bad_json"
    assert exc.value.raw_response == "still garbage"


def test_invalid_step_range_rejected_then_repaired(run1):
    frames, chart, manifest = run1
    # 90000 > checkpoint_step (80000) + window (5000): fails validation,
    # exercising the repair round with a real validator message.
    bad = model_json(step_range=(90000, 95000))
    mock = MockClient([as_response(bad), as_response(model_json())])
    result = analyze_run(frames, chart, bootstrap_log(1), manifest, client=mock)
    assert result["next_to_check"]["step_range"] == [70000, 75000]
    assert len(mock.calls) == 2


def test_three_run_chain_carries_state_forward():
    log = bootstrap_log(1)
    prior_claim = "no penalty at failure event (frames 12-13; no curve dip)"
    for n in (1, 2, 3):
        frames, chart, manifest = load_run(n)
        cumulative = model_json(confirmed=[prior_claim, f"seen again in run {n}"])
        mock = MockClient([as_response(cumulative)])
        log = analyze_run(frames, chart, log, manifest, client=mock)
        assert log["run"] == n
        if n > 1:  # previous state was embedded in the prompt
            prompt_text = "\n".join(p["text"] for p in mock.calls[0]["messages"][1]["content"]
                                    if p["type"] == "text")
            assert prior_claim in prompt_text
    assert prior_claim in log["confirmed"]  # cumulative, not delta
