# Stream 2 `analyze_run` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `analyze_run(frames, chart, hypothesis_log, manifest=None) -> dict` — one local-Gemma multimodal inference call that turns a training run's rollout frames + reward chart + telemetry into an updated failure-hypothesis JSON.

**Architecture:** Small single-purpose modules composed by `analyze.py`: `client.py` is the only network code (OpenAI-compatible chat completions, llama-server primary), `prompts.py` and `schema.py` are pure functions, `images.py` handles encoding/frame-selection/contact-sheet, `mock_client.py` makes everything below the HTTP line testable without a model. Spec: `docs/superpowers/specs/2026-07-04-analyze-run-design.md`.

**Tech Stack:** Python 3.12 (repo venv `.venv`), `requests`, `Pillow`, `pytest`. Real test fixtures: `stream1_simulation/outputs/run{1,2,3}/`.

## Global Constraints

- Return shape = root README "Data Contract"; no new top-level fields (citations embedded in claim strings).
- Output `confirmed`/`ruled_out` are the **complete cumulative state**, not deltas.
- Bookkeeping fields `run`, `next_to_check.run`, `timestamp` are stamped by code, never trusted from the model.
- Degraded 3-arg path (no manifest): `next_to_check` forced to `None`; `run` from `hypothesis_log["run"]`.
- `next_to_check.step_range` = TRAINING steps, 2 ascending non-negative ints, ≤ `checkpoint_step` + one window width; NOT clamped to the manifest window.
- Our authored prompt text (system prompt + ask template) contains no bug-specific words: forbidden substrings `drop`, `table` (case-insensitive).
- Inference: temperature 0.2 (0 on repair), `max_tokens` 1200, client timeout 120s, thinking mode not enabled.
- Repair round is text-only (no images resent), at most one round, then `AnalyzeRunError(kind="bad_json")`.
- JSON extraction takes the **last** fenced ```json block (fallback: last balanced raw object), never the first.
- Default 8 frames; the manifest's `drop_step` frame, first, and last always survive subsampling; frame↔sim-step mapping by filename basename, never list index.
- Env config: `GEMMA_BASE_URL` (default `http://localhost:8080/v1`), `GEMMA_MODEL` (default `local-gemma`), `GEMMA_N_FRAMES` (default `8`), `GEMMA_CONTACT_SHEET` (default `0`).
- Commits after every green test cycle; all commands run from repo root with `./.venv/bin/python`.

## File Structure

```
stream2_gemma_inference/
├── __init__.py       # exports analyze_run (Stream 3 imports from the package)
├── errors.py         # AnalyzeRunError(kind=timeout|backend|bad_json)
├── schema.py         # strip_reasoning, extract_json, validate
├── images.py         # encode_image, select_frames, frame_labels, make_contact_sheet
├── prompts.py        # SYSTEM_PROMPT, ASK_TEMPLATE, format_telemetry, build_messages, build_repair_messages
├── client.py         # GemmaClient (only module that touches the network)
├── mock_client.py    # MockClient for tests
├── analyze.py        # analyze_run() — composition + bookkeeping stamping
├── smoke.py          # manual live gate + timing (not CI)
├── requirements.txt  # modified: uncomment requests, add pillow + pytest
├── README.md         # modified: server launch, usage, cross-stream flags
└── tests/
    ├── conftest.py
    ├── test_errors.py
    ├── test_schema.py
    ├── test_images.py
    ├── test_prompts.py
    ├── test_client.py
    └── test_analyze.py
```

---

### Task 1: Package skeleton, dependencies, `errors.py`

**Files:**
- Create: `stream2_gemma_inference/__init__.py`, `stream2_gemma_inference/errors.py`, `stream2_gemma_inference/tests/__init__.py` (empty — makes `tests` importable so test modules can import conftest helpers), `stream2_gemma_inference/tests/conftest.py`, `stream2_gemma_inference/tests/test_errors.py`
- Modify: `stream2_gemma_inference/requirements.txt`

**Interfaces:**
- Produces: `AnalyzeRunError(kind: str, message: str, raw_response: str | None = None)` with attributes `.kind`, `.raw_response`; kinds are `"timeout" | "backend" | "bad_json"`. Test fixtures `load_run(n)`, `bootstrap_log(n)` in conftest used by Tasks 3–6.

- [ ] **Step 1: Replace requirements.txt**

```
# Stream 2 — local Gemma inference over an OpenAI-compatible endpoint
requests>=2.31.0
huggingface_hub>=0.20.0
numpy>=1.24.0
pillow>=10.0
pytest>=8.0
```

Install: `./.venv/bin/pip install -r stream2_gemma_inference/requirements.txt`

- [ ] **Step 2: Write conftest and failing error test**

`stream2_gemma_inference/tests/conftest.py`:

```python
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
```

`stream2_gemma_inference/tests/test_errors.py`:

```python
from stream2_gemma_inference.errors import AnalyzeRunError


def test_error_carries_kind_and_raw_response():
    err = AnalyzeRunError("bad_json", "no JSON found", raw_response="blah")
    assert err.kind == "bad_json"
    assert err.raw_response == "blah"
    assert "no JSON found" in str(err)


def test_raw_response_optional():
    err = AnalyzeRunError("timeout", "gave up after 120s")
    assert err.kind == "timeout"
    assert err.raw_response is None
```

- [ ] **Step 3: Run to verify failure**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stream2_gemma_inference'` (or errors module missing)

- [ ] **Step 4: Implement**

`stream2_gemma_inference/__init__.py`:

```python
"""Stream 2 — local Gemma multimodal inference for RL policy debugging."""
```

(`analyze_run` re-export is added in Task 6 when it exists.)

`stream2_gemma_inference/errors.py`:

```python
class AnalyzeRunError(Exception):
    """Raised when analyze_run cannot produce a valid hypothesis.

    kind: "timeout"  — inference call exceeded the client timeout
          "backend"  — server unreachable / HTTP error / model not loaded
          "bad_json" — model output failed validation even after one repair round
    Stream 3 can branch on .kind; .raw_response holds the model's last output.
    """

    def __init__(self, kind, message, raw_response=None):
        super().__init__(message)
        self.kind = kind
        self.raw_response = raw_response
```

- [ ] **Step 5: Run to verify pass**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_errors.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add stream2_gemma_inference
git commit -m "feat(stream2): package skeleton, deps, AnalyzeRunError"
```

---

### Task 2: `schema.py` — reasoning-stripping, last-JSON extraction, validation

**Files:**
- Create: `stream2_gemma_inference/schema.py`, `stream2_gemma_inference/tests/test_schema.py`

**Interfaces:**
- Produces:
  - `strip_reasoning(text: str) -> str`
  - `extract_json(text: str) -> dict` — raises `ValueError` if no parseable object
  - `validate(candidate: dict, manifest: dict | None) -> list[str]` — empty list = valid; model-authored fields only (`run`/`timestamp` are stamped later, not validated here)

- [ ] **Step 1: Write failing tests**

`stream2_gemma_inference/tests/test_schema.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on schema

- [ ] **Step 3: Implement**

`stream2_gemma_inference/schema.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_schema.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add stream2_gemma_inference/schema.py stream2_gemma_inference/tests/test_schema.py
git commit -m "feat(stream2): schema module - reasoning strip, last-JSON extraction, validation"
```

---

### Task 3: `images.py` — encoding, event-frame-pinned selection, labels, contact sheet

**Files:**
- Create: `stream2_gemma_inference/images.py`, `stream2_gemma_inference/tests/test_images.py`

**Interfaces:**
- Consumes: conftest `load_run` (Task 1).
- Produces:
  - `encode_image(path: str) -> str` — `data:image/png;base64,...` URI
  - `select_frames(frames: list[str], manifest: dict | None, n_frames: int) -> list[str]` — temporal order preserved; first, last, and the `drop_step` frame always survive
  - `frame_labels(frames: list[str], manifest: dict | None) -> list[str]` — `"frame {i} = sim step {s}"` by basename match, else `"frame {i}"`
  - `make_contact_sheet(frames: list[str], labels: list[str], out_path: str, cols: int = 4) -> str`

- [ ] **Step 1: Write failing tests**

`stream2_gemma_inference/tests/test_images.py`:

```python
import base64

from PIL import Image

from stream2_gemma_inference.images import (
    encode_image,
    frame_labels,
    make_contact_sheet,
    select_frames,
)


def test_encode_image_is_png_data_uri(run1):
    frames, _, _ = run1
    uri = encode_image(frames[0])
    assert uri.startswith("data:image/png;base64,")
    base64.b64decode(uri.split(",", 1)[1])  # decodes cleanly


def test_select_keeps_all_when_budget_covers(run1):
    frames, _, manifest = run1
    assert select_frames(frames, manifest, 15) == frames


def test_select_pins_drop_frame_for_any_budget(run1):
    frames, _, manifest = run1
    drop_file = next(
        rec["file"] for rec in manifest["frames"] if rec["sim_step"] == manifest["drop_step"]
    )
    for n in range(3, 15):
        chosen = select_frames(frames, manifest, n)
        assert len(chosen) == n
        basenames = [f.rsplit("/", 1)[-1] for f in chosen]
        assert drop_file.rsplit("/", 1)[-1] in basenames, f"drop frame lost at n={n}"
        assert chosen[0] == frames[0] and chosen[-1] == frames[-1]
        assert chosen == sorted(chosen, key=frames.index)  # temporal order


def test_select_without_manifest_keeps_endpoints(run1):
    frames, _, _ = run1
    chosen = select_frames(frames, None, 8)
    assert len(chosen) == 8 and chosen[0] == frames[0] and chosen[-1] == frames[-1]


def test_labels_use_sim_steps_from_manifest(run1):
    frames, _, manifest = run1
    labels = frame_labels(frames, manifest)
    assert labels[0] == "frame 0 = sim step 0"
    assert labels[13] == f"frame 13 = sim step {manifest['drop_step']}"


def test_labels_by_basename_not_index(run1):
    frames, _, manifest = run1
    labels = frame_labels(list(reversed(frames)), manifest)
    assert labels[0] == f"frame 0 = sim step {manifest['frames'][-1]['sim_step']}"


def test_labels_plain_without_manifest(run1):
    frames, _, _ = run1
    assert frame_labels(frames[:2], None) == ["frame 0", "frame 1"]


def test_contact_sheet_dimensions(run1, tmp_path):
    frames, _, manifest = run1
    labels = frame_labels(frames, manifest)
    out = make_contact_sheet(frames, labels, str(tmp_path / "sheet.png"), cols=4)
    sheet = Image.open(out)
    assert sheet.width == 4 * 320          # 4 tiles wide
    assert sheet.height == 4 * (320 + 22)  # 15 frames -> 4 rows, 22px label strip
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_images.py -v`
Expected: FAIL — ImportError on images

- [ ] **Step 3: Implement**

`stream2_gemma_inference/images.py`:

```python
"""Image plumbing: base64 data-URIs, event-frame-pinned subsampling, sim-step
labels (by filename basename, never list index), and the contact-sheet latency
lever (per-image visual-token cost is fixed regardless of pixel size, so
tiling N frames into one image is the real token cut)."""

import base64
import math
import os

from PIL import Image, ImageDraw

LABEL_STRIP_PX = 22


def encode_image(path):
    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _basename_to_sim_step(manifest):
    if not manifest:
        return {}
    return {
        os.path.basename(rec["file"]): rec["sim_step"] for rec in manifest.get("frames", [])
    }


def select_frames(frames, manifest, n_frames):
    """Subsample to n_frames, always keeping first, last, and the manifest's
    drop_step frame — uniform-with-endpoints subsampling provably deletes the
    single most evidence-bearing frame (index 13 of 15 in all three fixtures)."""
    n_frames = max(3, n_frames)
    if len(frames) <= n_frames:
        return list(frames)

    pinned = {0, len(frames) - 1}
    step_map = _basename_to_sim_step(manifest)
    if manifest and "drop_step" in manifest:
        for i, path in enumerate(frames):
            if step_map.get(os.path.basename(path)) == manifest["drop_step"]:
                pinned.add(i)
                break

    remaining = n_frames - len(pinned)
    candidates = [i for i in range(len(frames)) if i not in pinned]
    if remaining > 0 and candidates:
        stride = len(candidates) / remaining
        pinned.update(candidates[min(int(k * stride), len(candidates) - 1)] for k in range(remaining))
    return [frames[i] for i in sorted(pinned)]


def frame_labels(frames, manifest):
    step_map = _basename_to_sim_step(manifest)
    labels = []
    for i, path in enumerate(frames):
        step = step_map.get(os.path.basename(path))
        labels.append(f"frame {i} = sim step {step}" if step is not None else f"frame {i}")
    return labels


def make_contact_sheet(frames, labels, out_path, cols=4):
    tiles = [Image.open(p).convert("RGB") for p in frames]
    w, h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * w, rows * (h + LABEL_STRIP_PX)), "black")
    draw = ImageDraw.Draw(sheet)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x = (i % cols) * w
        y = (i // cols) * (h + LABEL_STRIP_PX)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + h + 4), label, fill="white")
    sheet.save(out_path)
    return out_path
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_images.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add stream2_gemma_inference/images.py stream2_gemma_inference/tests/test_images.py
git commit -m "feat(stream2): images module - encoding, drop-frame-pinned selection, contact sheet"
```

---

### Task 4: `prompts.py` — system prompt, telemetry block, message builders

**Files:**
- Create: `stream2_gemma_inference/prompts.py`, `stream2_gemma_inference/tests/test_prompts.py`

**Interfaces:**
- Consumes: `encode_image` (Task 3).
- Produces:
  - `SYSTEM_PROMPT: str`, `ASK_TEMPLATE: str`
  - `format_telemetry(manifest: dict | None) -> str`
  - `build_messages(frame_items: list[tuple[str, str]], chart: str, hypothesis_log: dict, manifest: dict | None) -> list[dict]` — `frame_items` = `(label, image_path)` pairs (a contact sheet is one pair)
  - `build_repair_messages(problems: list[str], prior_output: str) -> list[dict]` — text-only

- [ ] **Step 1: Write failing tests**

`stream2_gemma_inference/tests/test_prompts.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_prompts.py -v`
Expected: FAIL — ImportError on prompts

- [ ] **Step 3: Implement**

`stream2_gemma_inference/prompts.py`:

```python
"""Prompt assembly. Everything here is a pure function of its inputs.

Authored text (SYSTEM_PROMPT, ASK_TEMPLATE) is failure-agnostic — it never
names the planted bug (unit-enforced). The DATA may name events; facts belong
to the model, the causal chain is its work."""

import json

from .images import encode_image

SYSTEM_PROMPT = """You are an RL policy debugging assistant. You are given \
evidence from one evaluation rollout of a robotic-manipulation policy partway \
through training: sequential video frames, a per-step reward chart for that \
single episode, run telemetry, and the hypothesis log from earlier runs. Your \
job is to work out whether the reward function is misspecified and, if so, \
propose a concrete fix.

Evidence rules:
- The telemetry block is authoritative for WHAT THE NUMBERS ARE. Never \
re-derive numeric values from chart pixels.
- The frames are authoritative for WHAT PHYSICALLY HAPPENED.
- Neither source outranks the other on WHY. A disagreement between them is a \
finding to report, not noise to smooth over. Agreement that a run failed does \
not end the analysis: the key signature of a missing reward term is a failure \
event visible in the frames that produces no corresponding penalty in the \
per-step reward.

Your final answer must be a single JSON object in a fenced block, exactly this \
shape (bookkeeping fields like run ids and timestamps are added by the caller \
— do not include them):

```json
{
  "confirmed": ["<claim> (frames 3-4; total_reward=-12.0)"],
  "ruled_out": ["<claim> (frames 1-9; reward slope steady)"],
  "next_to_check": {"step_range": [40000, 45000], "reason": "<why this window>"},
  "proposed_reward_edit": "<one-line reward-function change>"
}
```
next_to_check and proposed_reward_edit may be null."""

ASK_TEMPLATE = """ANALYSIS PROTOCOL — four stages, in order. Keep stages 1-3 \
to at most 4 sentences each.

STAGE 1 — VIDEO ONLY: Using only the frames (ignore all numbers), describe \
what the policy does across the episode and note any failure events, citing \
frame numbers.

STAGE 2 — NUMBERS ONLY: Using only the telemetry block and the reward chart, \
describe how reward evolves within the episode and the run's outcome.

STAGE 3 — CROSS-CHECK: For each failure event from Stage 1, state whether the \
per-step reward visibly reacts to it. An event with no reward reaction is \
evidence of a missing reward term. Note any disagreement between the two \
accounts.

STAGE 4 — UPDATED HYPOTHESES: Output the complete updated hypothesis state as \
the FINAL fenced ```json block. confirmed/ruled_out must be the full \
cumulative state: carry forward prior items you still believe, add new ones. \
Every item must cite frame number(s) AND a telemetry or chart fact inside the \
claim string; a claim supported by only one source may not be confirmed — \
route it to next_to_check instead. next_to_check.step_range must use TRAINING \
steps. No text after the JSON block."""

_TIMELINE_NOTE = """Two distinct step scales appear below — do not mix them:
- SIM steps index this single evaluation episode (num_episode_steps, drop_step, frame labels).
- TRAINING steps index the whole training run (checkpoint_step, step_range). \
next_to_check.step_range must use TRAINING steps."""


def format_telemetry(manifest):
    if not manifest:
        return "RUN TELEMETRY: no telemetry provided for this run."
    fields = {k: v for k, v in manifest.items() if k not in ("frames", "reward_curve")}
    lines = "\n".join(f"  {k}: {json.dumps(v)}" for k, v in fields.items())
    return (
        "RUN TELEMETRY (ground truth — cite numbers from here):\n"
        + _TIMELINE_NOTE + "\n" + lines
    )


def _text(t):
    return {"type": "text", "text": t}


def _image(path):
    return {"type": "image_url", "image_url": {"url": encode_image(path)}}


def build_messages(frame_items, chart, hypothesis_log, manifest):
    """frame_items: (label, image_path) pairs — individual frames or a single
    contact-sheet pair. Order inside the user message: telemetry, prior log,
    chart, frames, THEN the staged ask (frames-before-ask counters anchoring)."""
    content = [
        _text(format_telemetry(manifest)),
        _text("PRIOR HYPOTHESIS LOG (from earlier runs):\n"
              + json.dumps(hypothesis_log, indent=2)),
        _text("REWARD CHART — per-step reward over this single evaluation episode:"),
        _image(chart),
    ]
    for label, path in frame_items:
        content.append(_text(label + ":"))
        content.append(_image(path))
    content.append(_text(ASK_TEMPLATE))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_repair_messages(problems, prior_output):
    """Text-only repair round — images are NOT resent (latency)."""
    return [
        {"role": "system",
         "content": "You repair malformed JSON. Return ONLY the corrected JSON "
                     "object in a fenced ```json block. No other text."},
        {"role": "user",
         "content": "Your previous response failed validation:\n- "
                     + "\n- ".join(problems)
                     + "\n\nPrevious response:\n" + prior_output
                     + "\n\nReturn the corrected JSON object now."},
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_prompts.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add stream2_gemma_inference/prompts.py stream2_gemma_inference/tests/test_prompts.py
git commit -m "feat(stream2): prompts module - failure-agnostic staged protocol, timeline-annotated telemetry"
```

---

### Task 5: `client.py` + `mock_client.py`

**Files:**
- Create: `stream2_gemma_inference/client.py`, `stream2_gemma_inference/mock_client.py`, `stream2_gemma_inference/tests/test_client.py`

**Interfaces:**
- Consumes: `AnalyzeRunError` (Task 1).
- Produces:
  - `GemmaClient(base_url=None, model=None, timeout=120.0, temperature=0.2, max_tokens=1200)` with `.chat(messages, temperature=None) -> str` and `.last_usage: dict | None`
  - `MockClient(responses: list[str])` with the same `.chat` signature and `.calls: list[dict]` recording `{"messages": ..., "temperature": ...}`

- [ ] **Step 1: Write failing tests**

`stream2_gemma_inference/tests/test_client.py`:

```python
import pytest
import requests

from stream2_gemma_inference.client import GemmaClient
from stream2_gemma_inference.errors import AnalyzeRunError
from stream2_gemma_inference.mock_client import MockClient

MSGS = [{"role": "user", "content": "hi"}]


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


def _payload(content="hello", usage=None):
    return {"choices": [{"message": {"content": content}}], "usage": usage or {"prompt_tokens": 7}}


def test_chat_returns_content_and_records_usage(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, body=json, timeout=timeout)
        return FakeResp(payload=_payload("the answer"))

    monkeypatch.setattr(requests, "post", fake_post)
    assert client.chat(MSGS) == "the answer"
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["body"]["temperature"] == 0.2 and captured["body"]["max_tokens"] == 1200
    assert captured["timeout"] == 120.0
    assert client.last_usage == {"prompt_tokens": 7}


def test_per_call_temperature_override(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    captured = {}
    monkeypatch.setattr(requests, "post",
                        lambda url, json=None, timeout=None: captured.update(body=json) or FakeResp(payload=_payload()))
    client.chat(MSGS, temperature=0)
    assert captured["body"]["temperature"] == 0


def test_timeout_maps_to_kind_timeout(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout()))
    with pytest.raises(AnalyzeRunError) as exc:
        client.chat(MSGS)
    assert exc.value.kind == "timeout"


def test_connection_error_maps_to_kind_backend(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError()))
    with pytest.raises(AnalyzeRunError) as exc:
        client.chat(MSGS)
    assert exc.value.kind == "backend"


def test_http_error_maps_to_kind_backend(monkeypatch):
    client = GemmaClient(base_url="http://x/v1", model="m")
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(status=500))
    with pytest.raises(AnalyzeRunError) as exc:
        client.chat(MSGS)
    assert exc.value.kind == "backend"


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("GEMMA_BASE_URL", "http://env-host:9999/v1")
    monkeypatch.setenv("GEMMA_MODEL", "env-model")
    client = GemmaClient()
    assert client.base_url == "http://env-host:9999/v1"
    assert client.model == "env-model"


def test_mock_client_replays_and_records():
    mock = MockClient(["one", "two"])
    assert mock.chat(MSGS) == "one"
    assert mock.chat(MSGS, temperature=0) == "two"
    assert mock.calls[1]["temperature"] == 0
    assert mock.calls[0]["messages"] == MSGS
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_client.py -v`
Expected: FAIL — ImportError on client

- [ ] **Step 3: Implement**

`stream2_gemma_inference/client.py`:

```python
"""The only module that touches the network. OpenAI-compatible chat
completions; llama-server is the primary backend (default port 8080), LM
Studio the fallback (set GEMMA_BASE_URL=http://localhost:1234/v1)."""

import os

import requests

from .errors import AnalyzeRunError

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "local-gemma"


class GemmaClient:
    def __init__(self, base_url=None, model=None, timeout=120.0,
                 temperature=0.2, max_tokens=1200):
        self.base_url = (base_url or os.environ.get("GEMMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage = None

    def chat(self, messages, temperature=None):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 json=body, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise AnalyzeRunError("timeout", f"inference exceeded {self.timeout}s")
        except requests.exceptions.RequestException as exc:
            raise AnalyzeRunError("backend", f"cannot reach Gemma server at {self.base_url}: {exc}")
        if resp.status_code >= 400:
            raise AnalyzeRunError("backend", f"Gemma server returned HTTP {resp.status_code}")
        data = resp.json()
        self.last_usage = data.get("usage")
        return data["choices"][0]["message"].get("content") or ""
```

`stream2_gemma_inference/mock_client.py`:

```python
"""Canned-response stand-in for GemmaClient — lets everything below the HTTP
line run in tests without a model."""


class MockClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.last_usage = None

    def chat(self, messages, temperature=None):
        self.calls.append({"messages": messages, "temperature": temperature})
        return self._responses.pop(0)
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_client.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add stream2_gemma_inference/client.py stream2_gemma_inference/mock_client.py stream2_gemma_inference/tests/test_client.py
git commit -m "feat(stream2): GemmaClient (OpenAI-compatible, typed errors) + MockClient"
```

---

### Task 6: `analyze.py` — composition, bookkeeping, repair, contract tests

**Files:**
- Create: `stream2_gemma_inference/analyze.py`, `stream2_gemma_inference/tests/test_analyze.py`
- Modify: `stream2_gemma_inference/__init__.py`

**Interfaces:**
- Consumes: everything above — `select_frames`, `frame_labels`, `make_contact_sheet`, `build_messages`, `build_repair_messages`, `extract_json`, `validate`, `GemmaClient`/`MockClient`, `AnalyzeRunError`.
- Produces: `analyze_run(frames, chart, hypothesis_log, manifest=None, *, client=None, n_frames=None, contact_sheet=None) -> dict`, re-exported from the package (`from stream2_gemma_inference import analyze_run` — this is Stream 3's import).

- [ ] **Step 1: Write failing tests**

`stream2_gemma_inference/tests/test_analyze.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests/test_analyze.py -v`
Expected: FAIL — cannot import `analyze_run`

- [ ] **Step 3: Implement**

`stream2_gemma_inference/analyze.py`:

```python
"""analyze_run — Stream 2's single deliverable. Composes images/prompts/client/
schema; owns bookkeeping stamping and the one text-only repair round."""

import os
import tempfile
from datetime import datetime, timezone

from .client import GemmaClient
from .errors import AnalyzeRunError
from .images import frame_labels, make_contact_sheet, select_frames
from .prompts import build_messages, build_repair_messages
from .schema import extract_json, validate

DEFAULT_N_FRAMES = 8


def _resolve(value, env_key, default, cast):
    if value is not None:
        return value
    return cast(os.environ.get(env_key, default))


def _frame_items(frames, manifest, n_frames, contact_sheet):
    chosen = select_frames(frames, manifest, n_frames)
    labels = frame_labels(chosen, manifest)
    if not contact_sheet:
        return list(zip(labels, chosen))
    sheet = os.path.join(tempfile.mkdtemp(prefix="stream2_"), "contact_sheet.png")
    make_contact_sheet(chosen, labels, sheet)
    return [(f"CONTACT SHEET of {len(chosen)} rollout frames "
             "(per-tile labels inside the image)", sheet)]


def _parse_or_problems(raw, manifest):
    try:
        candidate = extract_json(raw)
    except ValueError as exc:
        return None, [str(exc)]
    problems = validate(candidate, manifest)
    return (candidate, problems)


def analyze_run(frames, chart, hypothesis_log, manifest=None, *,
                client=None, n_frames=None, contact_sheet=None):
    """Analyze one training run's evidence; return the updated hypothesis entry
    (root README data-contract shape, cumulative state — persist verbatim)."""
    client = client or GemmaClient()
    n_frames = _resolve(n_frames, "GEMMA_N_FRAMES", DEFAULT_N_FRAMES, int)
    contact_sheet = _resolve(contact_sheet, "GEMMA_CONTACT_SHEET", "0",
                             lambda v: str(v) not in ("0", "false", "False", ""))

    items = _frame_items(frames, manifest, n_frames, contact_sheet)
    raw = client.chat(build_messages(items, chart, hypothesis_log, manifest))
    candidate, problems = _parse_or_problems(raw, manifest)

    if problems:  # one text-only repair round at temperature 0
        raw = client.chat(build_repair_messages(problems, raw), temperature=0)
        candidate, problems = _parse_or_problems(raw, manifest)
        if problems:
            raise AnalyzeRunError(
                "bad_json", f"model output failed validation after repair: {problems}",
                raw_response=raw)

    # Bookkeeping is ours, never the model's.
    run = manifest["run"] if manifest else hypothesis_log.get("run")
    candidate["run"] = run
    candidate["timestamp"] = datetime.now(timezone.utc).isoformat()
    if manifest is None:
        # Without telemetry the model has no training-step vocabulary; any
        # range it emitted would send Stream 4's browser somewhere meaningless.
        candidate["next_to_check"] = None
    elif candidate.get("next_to_check"):
        candidate["next_to_check"]["run"] = run
    return candidate
```

Update `stream2_gemma_inference/__init__.py`:

```python
"""Stream 2 — local Gemma multimodal inference for RL policy debugging."""

from .analyze import analyze_run

__all__ = ["analyze_run"]
```

- [ ] **Step 4: Run full suite**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests -v`
Expected: all tests pass (Tasks 1–6: ~47 tests)

- [ ] **Step 5: Commit**

```bash
git add stream2_gemma_inference/analyze.py stream2_gemma_inference/__init__.py stream2_gemma_inference/tests/test_analyze.py
git commit -m "feat(stream2): analyze_run - composition, bookkeeping stamping, repair round, contract tests"
```

---

### Task 7: `smoke.py` live gate + README

**Files:**
- Create: `stream2_gemma_inference/smoke.py`
- Modify: `stream2_gemma_inference/README.md`

**Interfaces:**
- Consumes: `analyze_run`, `GemmaClient`, conftest-style run loading (duplicated locally — smoke must run without pytest).
- Produces: `python -m stream2_gemma_inference.smoke --run_id 1 [--gate-only]` — the hour-0 two-image vision gate, then one timed real call. Manual only, never CI.

- [ ] **Step 1: Implement smoke script** (no unit test — this IS the manual test; the deliverable is reviewed by running it against a live server)

`stream2_gemma_inference/smoke.py`:

```python
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
```

- [ ] **Step 2: Verify it degrades cleanly with no server**

Run: `./.venv/bin/python -m stream2_gemma_inference.smoke --gate-only`
Expected: `AnalyzeRunError` with kind `backend` (connection refused) — confirms error mapping end-to-end. (With a live server this prints the gate result instead.)

- [ ] **Step 3: Replace README**

`stream2_gemma_inference/README.md`:

```markdown
# Stream 2 — Gemma 12B Inference Pipeline

*Owner: RL*

**Deliverable (live):**

```python
from stream2_gemma_inference import analyze_run

entry = analyze_run(frames, chart, hypothesis_log, manifest)  # manifest: pass it!
```

Design: `docs/superpowers/specs/2026-07-04-analyze-run-design.md`.

## Server (start here, hour 0)

Primary backend is llama.cpp's `llama-server` (prebuilt, reliable Gemma
vision); LM Studio is the fallback if it loads the model's vision path:

```bash
brew install llama.cpp
llama-server -hf <team's gemma-12b vision GGUF, e.g. ggml-org/gemma-3-12b-it-GGUF> \
             -c 16384 --port 8080
# then ALWAYS run the vision gate before building on it:
python -m stream2_gemma_inference.smoke --run_id 1 --gate-only
```

Context ≥ 8k is required (15 frames + chart ≈ 4.5k visual tokens). Config via
env: `GEMMA_BASE_URL` (default `http://localhost:8080/v1`), `GEMMA_MODEL`,
`GEMMA_N_FRAMES` (default 8; the drop/event frame always survives
subsampling), `GEMMA_CONTACT_SHEET=1` (first latency lever).

## For Stream 3 (cross-stream contract)

- Pass `manifest` (the run's `manifest.json` dict) as the 4th argument — you
  already load it for the Computer Use trigger. Without it,
  `next_to_check` is always `null` and `run` is stamped from the incoming log.
- Input `hypothesis_log` = the previous run's full entry (run 1: the empty
  bootstrap). The RETURN is the complete cumulative state — persist it
  verbatim, no merging.
- On failure, `AnalyzeRunError` with `.kind` ∈ `timeout | backend | bad_json`
  (`.raw_response` attached). Malformed data is never returned.
- Claim strings embed their evidence citations, e.g.
  `"no penalty at failure event (frames 12-13; drop_step=65, no dip)"`.

## Tests

```bash
./.venv/bin/python -m pytest stream2_gemma_inference/tests -v   # no model needed
python -m stream2_gemma_inference.smoke --run_id 1              # live, model needed
```
```

- [ ] **Step 4: Run full suite one last time**

Run: `./.venv/bin/python -m pytest stream2_gemma_inference/tests -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add stream2_gemma_inference/smoke.py stream2_gemma_inference/README.md
git commit -m "feat(stream2): live smoke gate + README with server launch and Stream 3 contract notes"
```

---

## Plan Self-Review Notes

- **Spec coverage:** multi-signal input (T4/T6), missing-penalty signature + equal-weight + staged protocol + dual citation (T4), two-timeline annotation (T4), last-JSON extraction + loose training-step validation (T2), degraded path forcing `next_to_check=null` + run stamping (T6), drop-frame pinning + contact sheet + env knobs (T3/T6), llama-server-first + hour-0 gate + typed errors + text-only repair + temp/max_tokens (T5/T6/T7), chained run1→2→3 contract test (T6), requirements fix (T1), cross-stream flags documented (T7). Latency measurement = running `smoke.py` at the hour-6 checkpoint.
- **Deliberately deferred (YAGNI):** GBNF/`response_format` escape hatch (spec lists it as non-default), visual-token-budget knob (not settable server-side), thinking-mode enablement (off per spec).
```
