"""Hypothesis-JSON handling: strip reasoning, extract the LAST JSON object,
validate the model-authored fields (bookkeeping fields are stamped in analyze.py)."""

import json
import re

_REASONING_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL),
]

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

REQUIRED_KEYS = ("confirmed", "ruled_out", "next_to_check", "proposed_reward_edit")


def strip_reasoning(text):
    for pat in _REASONING_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def _balanced_objects(text):
    """All top-level {...} substrings, in order."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                objs.append(text[start : i + 1])
    return objs


def extract_json(text):
    """Parse the LAST fenced ```json block; fall back to the last balanced raw
    object. Last, not first: the staged prompt makes the model echo our own
    telemetry JSON early in its response."""
    text = strip_reasoning(text)
    candidates = [m.group(1) for m in _FENCED.finditer(text)] or _balanced_objects(text)
    for raw in reversed(candidates):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no parseable JSON object in model response")


def _check_step_range(rng, manifest, problems):
    if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, int) for x in rng)):
        problems.append("next_to_check.step_range must be a list of 2 ints")
        return
    lo, hi = rng
    if lo < 0 or lo >= hi:
        problems.append("next_to_check.step_range must be ascending and non-negative")
    if manifest:
        window = manifest["step_range"][1] - manifest["step_range"][0]
        cap = manifest["checkpoint_step"] + window
        if hi > cap:
            problems.append(
                f"next_to_check.step_range exceeds plausible training steps (max {cap}); "
                "use TRAINING steps, not sim steps"
            )


def validate(candidate, manifest):
    """Return list of problems (empty = valid). Written to be echoed verbatim
    into the repair prompt."""
    problems = []
    if not isinstance(candidate, dict):
        return ["output must be a JSON object"]
    for key in REQUIRED_KEYS:
        if key not in candidate:
            problems.append(f"missing required key: {key}")
    for key in ("confirmed", "ruled_out"):
        val = candidate.get(key)
        if key in candidate and not (
            isinstance(val, list) and all(isinstance(item, str) for item in val)
        ):
            problems.append(f"{key} must be a list of strings")
    ntc = candidate.get("next_to_check")
    if ntc is not None and "next_to_check" in candidate:
        if not isinstance(ntc, dict):
            problems.append("next_to_check must be an object or null")
        else:
            _check_step_range(ntc.get("step_range"), manifest, problems)
            if not (isinstance(ntc.get("reason"), str) and ntc.get("reason").strip()):
                problems.append("next_to_check.reason must be a non-empty string")
    pre = candidate.get("proposed_reward_edit")
    if "proposed_reward_edit" in candidate and pre is not None and not isinstance(pre, str):
        problems.append("proposed_reward_edit must be a string or null")
    return problems
