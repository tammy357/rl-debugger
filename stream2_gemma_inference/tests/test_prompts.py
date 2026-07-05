import json

from stream2_gemma_inference.images import frame_labels
from stream2_gemma_inference.prompts import (
    ASK_TEMPLATE,
    SYSTEM_PROMPT,
    build_messages,
    build_repair_messages,
    format_telemetry,
)

FORBIDDEN = ("drop", "table")  # our authored text stays failure-agnostic


def _texts(message):
    return "\n".join(p["text"] for p in message["content"] if p["type"] == "text")


def _images(message):
    return [p for p in message["content"] if p["type"] == "image_url"]


def _messages(run, log=None):
    frames, chart, manifest = run
    items = list(zip(frame_labels(frames, manifest), frames))
    return build_messages(items, chart, log or {"run": 1, "confirmed": [], "ruled_out": [],
                                                "next_to_check": None, "proposed_reward_edit": None},
                          manifest)


def test_authored_text_is_failure_agnostic():
    for word in FORBIDDEN:
        assert word not in SYSTEM_PROMPT.lower()
        assert word not in ASK_TEMPLATE.lower()


def test_system_prompt_has_contract_essentials():
    assert "```json" in SYSTEM_PROMPT                # schema example present
    assert "never re-derive" in SYSTEM_PROMPT.lower()  # equal-weight rule
    assert "no corresponding penalty" in SYSTEM_PROMPT.lower()  # missing-penalty signature


def test_ask_has_staged_protocol_and_citations():
    for marker in ("STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4"):
        assert marker in ASK_TEMPLATE
    assert "TRAINING steps" in ASK_TEMPLATE
    assert "cite" in ASK_TEMPLATE.lower()


def test_telemetry_annotates_both_timelines(run1):
    _, _, manifest = run1
    block = format_telemetry(manifest)
    assert "SIM steps" in block and "TRAINING steps" in block
    assert "total_reward" in block and "drop_step" in block  # unfiltered fields
    assert str(manifest["checkpoint_step"]) in block


def test_messages_structure(run1):
    msgs = _messages(run1)
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]
    assert len(_images(user)) == 16  # 15 frames + 1 chart
    text = _texts(user)
    assert "PRIOR HYPOTHESIS LOG" in text and "frame 13 = sim step 65" in text
    assert text.rstrip().endswith(ASK_TEMPLATE.rstrip())  # ask comes last


def test_prior_log_is_embedded_verbatim(run1):
    log = {"run": 1, "confirmed": ["a prior finding"], "ruled_out": [],
           "next_to_check": None, "proposed_reward_edit": None}
    assert "a prior finding" in _texts(_messages(run1, log)[1])


def test_no_manifest_message_says_so(run1):
    frames, chart, _ = run1
    items = [(f"frame {i}", f) for i, f in enumerate(frames)]
    msgs = build_messages(items, chart, {"run": 1, "confirmed": [], "ruled_out": [],
                                         "next_to_check": None, "proposed_reward_edit": None}, None)
    assert "no telemetry" in _texts(msgs[1]).lower()


def test_repair_messages_are_text_only():
    msgs = build_repair_messages(["missing required key: ruled_out"], "prior blob")
    assert all(isinstance(m["content"], str) for m in msgs)
    joined = "\n".join(m["content"] for m in msgs)
    assert "missing required key: ruled_out" in joined and "prior blob" in joined
    assert "```json" in joined  # demands fenced corrected output


def test_stage3_checks_outcome_fields_against_frames():
    # run96 lesson: a lying success flag must be treated as a finding, not
    # trusted. Stage 3 must explicitly demand the outcome-vs-frames check.
    assert "outcome fields" in ASK_TEMPLATE
    assert "contradict" in ASK_TEMPLATE.lower()
