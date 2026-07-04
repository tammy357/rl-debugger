# Stream 2 — `analyze_run` Design

*Date: 2026-07-04 · Branch: `stream-2/gemma-inference` · Status: approved in discussion, pending spec review*

## Goal

Deliver Stream 2's single deliverable: a callable function that takes one
training run's evidence and returns an updated failure-hypothesis JSON, using
local Gemma 12B multimodal inference. Stream 3 imports it directly into the
agent loop. Target: one full call completes in under 60 seconds.

## Key design decision: multi-signal input

Video frames alone are unreliable evidence — Gemma may misread them, and a
chart PNG still requires visual reasoning to extract numbers we already have.
So `analyze_run` receives **three complementary signals plus memory**:

| Signal | Form | Job |
|---|---|---|
| Rollout frames | 15 images, 320×320 RGB | General-purpose sensor: shows what the policy *actually does*, including loopholes nobody pre-built a detector for |
| Reward curve | 1 chart image | Visual trend over the training window |
| Run telemetry (`manifest.json`) | Structured text | Hard facts: `total_reward`, `success`, `num_episode_steps`, `drop_step`, `step_range`, frame→sim-step mapping |
| Hypothesis log | Structured text | Memory: what previous runs confirmed / ruled out |

Rationale (agreed 2026-07-04): logs can only answer questions someone thought
to ask; the video is the only sensor for *unanticipated* exploit behavior.
Conversely, the logs anchor facts the model can't misread. The cross-check is
the core detection mechanism: **"reward is high (log) but the task visibly
failed (video)" is the signature of reward hacking** — the contradiction
between signals *is* the finding. Feeding both does not spoil the demo: the
causal inference (behavior → missing reward term → proposed edit) is never in
the logs and remains entirely the model's work.

**Consequence — failure-agnostic prompting.** The system prompt asks Gemma to
find mismatches between what the reward claims and what the video shows. It
never mentions dropping, tables, or any specific bug. The planted drop bug is
what it should find in this demo, but the tool stays general.

## Interface

```python
def analyze_run(
    frames: list[str],           # paths to rollout frames, temporal order
    chart: str,                  # path to reward-curve PNG
    hypothesis_log: dict,        # current state per README data contract
    manifest: dict | None = None # run telemetry (stream1 manifest.json)
) -> dict:                       # updated hypothesis, README data-contract shape
```

`manifest` is a **new, optional 4th parameter** — optional so Stream 3's
existing 3-argument call keeps working unchanged (degrades to video+chart-only
analysis). Flag the addition to Stream 3's owner so the agent loop passes it;
the loop already reads `manifest.json` for the Computer Use trigger.

Return shape (root README "Data Contract"):

```json
{
  "run": 2,
  "timestamp": "…",
  "confirmed": ["…"],
  "ruled_out": ["…"],
  "next_to_check": {"run": 2, "step_range": [47000, 52000], "reason": "…"},
  "proposed_reward_edit": "…"
}
```

## Architecture

```
stream2_gemma_inference/
├── analyze.py      # analyze_run() — the only public entry point
├── client.py       # GemmaClient: HTTP to an OpenAI-compatible local server
├── prompts.py      # system prompt + user-message builder (multimodal)
├── schema.py       # hypothesis JSON schema + validate() + extract_json()
├── images.py       # load path → base64 data-URI, optional downscale
├── mock_client.py  # canned-response client for tests (no model needed)
└── tests/
```

Each module has one job and is testable alone: `prompts` and `schema` are pure
functions; `client` is the only thing that touches the network; `analyze.py`
composes them.

### Backend: OpenAI-compatible local server (LM Studio first)

Decision: talk to Gemma over a local **OpenAI-compatible chat-completions
endpoint** (`base_url` from env `GEMMA_BASE_URL`, default
`http://localhost:1234/v1`), sending images as base64 data-URIs.

- **Primary: LM Studio** — GUI model download of the pre-quantized Gemma GGUF,
  built-in vision support, zero build step on Apple Silicon. Matches the
  "don't overthink hour 0–4" plan.
- Because the client is plain HTTP against a standard API, `llama.cpp`'s
  `llama-server` or Ollama are drop-in fallbacks via config — no code change.
- Dependency cost: just `requests` (+ `Pillow` for image prep). Avoids
  compiling `llama-cpp-python` locally (Stream 1's README documents how
  painful native builds are on this machine).

### Prompt strategy

- System prompt: role ("RL policy debugging assistant"), the failure-agnostic
  cross-check instruction, the exact output JSON schema with one example, and
  thinking mode enabled via `<|think|>` (per build plan).
- User message, in order: (1) telemetry block as compact text ("only cite
  numbers from here"), (2) prior hypothesis log, (3) reward chart image,
  (4) frames interleaved with `frame N = sim step S` labels from the manifest
  (plain `frame N` labels when no manifest is given),
  (5) the ask: describe observed behavior → cross-check against telemetry →
  update confirmed/ruled_out → pick `next_to_check` (must lie inside the
  manifest's `step_range`) → propose a reward edit. Every claim must cite the
  supporting frame numbers — proves the model actually looked.

### Output handling

1. Strip any `<think>…</think>`/thinking block, then extract the first JSON
   object (fenced or raw) from the response.
2. Validate against the schema (required keys, types, `step_range` a 2-int
   list within the manifest window when manifest present).
3. On failure: one repair round-trip — re-prompt with the validation error and
   "return only corrected JSON". If that fails, raise `AnalyzeRunError` with
   the raw response attached (Stream 3 decides retry/skip; we never return
   malformed data downstream).
4. Fill `run` (from manifest, else hypothesis_log) and `timestamp` ourselves —
   never trust the model with bookkeeping fields.

### Performance knobs (under-60s target)

Frame count (15 → 8 by uniform subsampling, always keeping first/last) and
image downscale are keyword options on `analyze_run`, defaulted from env, per
the hour-18–24 plan (visual token budget 560 → 280). Measure first; only turn
knobs if slow.

## Testing

- **Unit (no model):** `schema.py` accept/reject cases incl. JSON embedded in
  prose and thinking blocks; `prompts.py` message structure (telemetry present,
  frames labeled, no bug-specific words like "drop" in the system prompt);
  `images.py` encoding.
- **Contract:** `analyze_run` with `mock_client` against the real
  `stream1_simulation/outputs/run1..3` fixtures — verifies we consume exactly
  what `check_gemma_contract.py` produces and return the README shape,
  including the 3-arg (no manifest) call path.
- **Repair path:** mock client returns malformed JSON first, valid JSON on
  retry; and malformed twice → `AnalyzeRunError`.
- **Live smoke (manual, model running):** `python -m
  stream2_gemma_inference.smoke --run_id 1` — one real call, prints latency +
  validated JSON. Not in CI.

## Risks / open items

- **Cross-stream contract change** (4th param) — communicate to Stream 3.
- **Gemma JSON discipline** — mitigated by schema-in-prompt, extraction
  tolerant of prose, one repair round; hours 12–18 are reserved for prompt
  hardening against real outputs.
- **Latency unknown** until the model runs locally; knobs exist.
- **Model naming**: build plan says "Gemma 4 12B"; use whatever pre-quantized
  12B vision GGUF the team's LM Studio actually serves — client takes the
  model name from config.
