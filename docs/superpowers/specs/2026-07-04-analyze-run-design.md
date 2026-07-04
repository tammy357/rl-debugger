# Stream 2 — `analyze_run` Design

*Date: 2026-07-04 · Branch: `stream-2/gemma-inference` · Status: revised after
adversarial review (2 independent reviewers), pending user review*

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
| Rollout frames | up to 15 images, 320×320 RGB (14 sampled + the event frame Stream 1 forces in) | General-purpose sensor: shows what the policy *actually does*, including loopholes nobody pre-built a detector for |
| Reward curve | 1 chart image — **per-step reward over this single evaluation episode** (x = rollout step 0..~66), not the training window | Shows *when* within the episode reward flows, and what events coincide |
| Run telemetry (`manifest.json`) | Structured text | Hard facts: `total_reward`, `success`, `num_episode_steps`, `drop_step`, `checkpoint_step`, `step_range`, frame→sim-step mapping |
| Hypothesis log | Structured text | Memory: what previous runs confirmed / ruled out |

Rationale (agreed 2026-07-04): logs can only answer questions someone thought
to ask; the video is the only sensor for *unanticipated* exploit behavior.
Conversely, the logs anchor facts the model can't misread.

### The detection signature (corrected after review)

The naive reward-hacking signature — "reward is high but the task visibly
failed" — **does not occur in our actual data**: all three runs have negative
`total_reward` and `success: false`, so telemetry and video *agree* the runs
failed. The signature that actually exists is a **missing-penalty signature**:

> A salient failure event visible in the frames (and locatable on the
> episode's reward curve) that produces **no corresponding penalty** in the
> per-step reward — the curve doesn't dip when the failure happens.

The cross-check instruction is written to this signature: locate failure
events in the video, then check whether the reward *reacted* to them. The
"agreement is not exoneration" principle still holds — a failed run whose
reward never punishes the visible failure is precisely the evidence of a
missing reward term.

**Consequence — failure-agnostic prompting.** The system prompt and the ask
template ask Gemma to find failure events and un-punished behavior generally;
*our authored instructions* never mention dropping, tables, or any specific
bug (unit-tested on the assembled system prompt + ask template). The *data*
may name events (see "Known data leak" under Risks) — facts belong to the
model; the causal chain to the diagnosis and the reward edit is its work.

## Interface

```python
def analyze_run(
    frames: list[str],           # paths to rollout frames, temporal order
    chart: str,                  # path to reward-curve PNG
    hypothesis_log: dict,        # PREVIOUS run's full entry (see semantics)
    manifest: dict | None = None # run telemetry (stream1 manifest.json)
) -> dict:                       # updated hypothesis entry, README contract shape
```

### `hypothesis_log` semantics (pinned)

- **Input** is the previous run's complete entry (README "Data Contract"
  shape). On run 1 it's the empty bootstrap Stream 3 creates —
  `confirmed: []`, `ruled_out: []`, `next_to_check: None`,
  `proposed_reward_edit: None` (exactly the shape
  `check_gemma_contract.py` passes); `timestamp` may be absent on input.
- **Output is the complete updated state, not a delta**: `confirmed` /
  `ruled_out` carry forward prior items (minus any the model explicitly
  revises) plus new ones. Stream 3 persists the return verbatim — no merging
  on their side. (Stream 4's UI renders the latest entry and diff-flags new
  items, so deltas would make run-1 findings vanish on run 2.)
- Contract-tested across a simulated run1→run2→run3 chain, not just single
  calls.

### `manifest` parameter and the degraded path

`manifest` is optional only for call-compatibility with the current 3-arg
contract. **Stream 3 should always pass it** (the loop already loads
`manifest.json` for the Computer Use trigger). Degraded 3-arg behavior is
pinned so garbage can never reach Stream 4:

- `next_to_check` is forced to `null` — without the manifest the model has no
  training-step vocabulary, so any range it invented would be sim-steps or
  chart-pixel guesses that Computer Use would navigate WandB to meaninglessly.
- `run` is stamped from `hypothesis_log["run"]` (documented limitation: this
  is the previous run's number; another reason Stream 3 should pass the
  manifest, from which `run` is stamped authoritatively).
- Frame labels fall back from `frame N = sim step S` to plain `frame N`.

### Bookkeeping fields (never trusted to the model)

`run`, `next_to_check.run`, and `timestamp` are filled by our code — `run`s
from the manifest, timestamp from the clock. `schema.py` validates
`next_to_check.run` matches.

### Return shape

Root README "Data Contract" — unchanged, no cross-stream schema change:

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

Citations are embedded in the claim strings by convention —
`"no penalty at failure event (frames 12–13; drop_step=65, no dip on curve)"`
— giving Stream 4 provenance to render without a schema change.

## Architecture

```
stream2_gemma_inference/
├── analyze.py      # analyze_run() — the only public entry point
├── client.py       # GemmaClient: HTTP to an OpenAI-compatible local server
├── prompts.py      # system prompt + user-message builder (multimodal)
├── schema.py       # hypothesis JSON schema + validate() + extract_json()
├── images.py       # load path → base64 data-URI; frame pinning/subsampling; contact sheet
├── mock_client.py  # canned-response client for tests (no model needed)
└── tests/
```

Each module has one job and is testable alone: `prompts` and `schema` are pure
functions; `client` is the only thing that touches the network; `analyze.py`
composes them.

### Backend: OpenAI-compatible local server (llama-server first)

Talk to Gemma over a local **OpenAI-compatible chat-completions endpoint**
(`GEMMA_BASE_URL` env, model name from config), images as base64 data-URIs.
Multi-image requests per message are supported by both candidate servers; the
binding constraints are token costs, not the API shape.

- **Primary: `llama.cpp`'s `llama-server`** — prebuilt binary (brew install,
  no native build; the painful-compile concern applies to `llama-cpp-python`,
  which we don't use), auto-fetches model + vision projector via `-hf`, and is
  currently the more reliable path for Gemma vision GGUFs (LM Studio has
  reported mmproj-loading failures for exactly this model family).
- **Fallback: LM Studio** if it loads the model's vision path on the demo
  machine — same client code, different base URL.
- **Hour-0 gate (before building anything on the backend):** send a 2-image
  request; the model must describe *both* images. Catches silently-text-only
  vision setups immediately.
- Server launch config is part of the deliverable: **context ≥ 8k (16k
  comfortable)** — 15 frames + chart ≈ 4.5k visual tokens before any text, so
  a 4k default silently truncates — plus the chosen model/projector flags.
  The smoke test asserts `usage.prompt_tokens` is in the expected band.
- Dependencies: `requests` + `Pillow` (uncomment/add in
  `stream2_gemma_inference/requirements.txt`).

### Inference-call discipline

- `temperature 0.2` (0 on the repair round), explicit `max_tokens` (~1200).
- **Thinking mode off by default.** The build plan's "`<|think|>` in the
  system prompt" is not a reliable activation mechanism over an
  OpenAI-compatible endpoint (template/token handling varies by server and
  can silently degrade to literal text); if we enable thinking at all it's
  via the server's chat-template kwargs, treated as a quality knob to turn
  *up* only after latency is measured — an uncapped thinking budget alone
  can blow the 60s target.
- Hard client timeout (default 120s) → `AnalyzeRunError(kind="timeout")`;
  connection-refused/model-not-loaded → `kind="backend"`; malformed output
  after repair → `kind="bad_json"` with the raw response attached. Stream 3
  can react differently per kind; we never return malformed data downstream.

### Prompt strategy

- System prompt: role ("RL policy debugging assistant"), the failure-agnostic
  missing-penalty cross-check instruction, the exact output JSON schema with
  one example.
- **Two timelines, explicitly annotated** — the prompt states: sim steps
  (`drop_step`, `num_episode_steps`, frame labels) index *this one ~66-step
  evaluation episode*; training steps (`checkpoint_step`, `step_range`) index
  the *full training run*; `next_to_check.step_range` MUST be training steps.
  Without this the model sees "frame 13 = sim step 65" next to
  "step_range [77500, 82500]" with no way to reconcile them.
- **Equal-weight evidence rule**: telemetry is authoritative for *what the
  numbers are* (never re-derive them from the chart), the frames are
  authoritative for *what physically happened* — and **neither outranks the
  other on *why***. A disagreement between them is a finding to report, not
  noise to smooth over; and agreement that a run failed does not end the
  analysis — the question is whether the reward *reacted* to each failure
  event.
- **Staged analysis protocol** (anchoring prevention), each stage limited to
  a few sentences:
  1. *Video-only account* — the policy's behavior across the frames, before
     consulting any numbers.
  2. *Telemetry + chart account* — what the numbers and curve say, on their
     own terms.
  3. *Cross-check* — for each failure event from stage 1: does the reward
     visibly penalize it (curve dip / reward drop at that step)? Un-punished
     failure events are the primary evidence of a missing reward term.
  4. *Update hypotheses* — confirmed / ruled_out (complete updated state) /
     `next_to_check` (training steps) / proposed reward edit — emitted as the
     **final fenced ```json block**.
- **Dual citation**: each confirmed/ruled_out item cites frame number(s) and
  the telemetry or chart fact supporting it, embedded in the claim string;
  single-source claims must be labeled as such and belong in `next_to_check`
  rather than `confirmed`.
- User message, in order: (1) telemetry block — all manifest fields, with the
  timeline annotations above, (2) prior hypothesis log, (3) reward chart
  image, (4) frames interleaved with `frame N = sim step S` labels (plain
  `frame N` without manifest), (5) the staged ask. Frames-before-numbers
  ordering inside the ask counters anchoring on whichever evidence was read
  first.

### Output handling

1. Strip any thinking/reasoning content — prefer the response's
   `reasoning_content` field when present; otherwise strip known thought
   delimiters. Extraction must tolerate reasoning arriving as plain prose.
2. Extract the **last** fenced ```json block (fallback: last raw JSON
   object). *Not the first* — the staged protocol has the model summarizing
   the manifest, so an early JSON-ish echo of our own input is likely.
3. Validate against the schema: required keys, types;
   `next_to_check.step_range` a 2-int ascending list of plausible training
   steps (non-negative, ≤ `checkpoint_step` + one window) — **not** clamped
   to the manifest's `step_range`, since re-examining *earlier* training is a
   legitimate ask (the README's own contract example points outside the
   current window).
4. On failure: one **text-only** repair round-trip (validation error + the
   model's prior output; images are NOT resent) at temperature 0. If that
   fails, raise `AnalyzeRunError(kind="bad_json")`.
5. Stamp bookkeeping fields (`run`, `next_to_check.run`, `timestamp`)
   ourselves per Interface section.

### Performance (under-60s target)

Measured at the **hour-6 integration checkpoint**, not hour 18 — the risk
profile (12B vision model on Apple Silicon) makes latency the likeliest
blocker, and discovering it late leaves no room to react. Defaults start
conservative; quality knobs turn *up* if there's headroom:

- **Default: 8 frames** — subsampled with the manifest's event frame
  (`drop_step`) **pinned**, plus first and last, remainder uniform.
  Uniform-with-endpoints subsampling would deterministically discard the drop
  frame (index 13 of 15 in all three fixtures) — the single most
  evidence-bearing image. Unit-tested: the event frame survives any
  `n_frames`. Frame-label mapping is by filename against
  `manifest.frames[].file`, never by list index, so subsampling and
  caller-reordered lists can't mislabel.
- **First latency lever: contact sheet** — tile all frames into 1–2 labeled
  grid images (`images.py`). Per-image cost is a fixed processor-side token
  budget regardless of pixel dimensions, so downscaling individual frames
  saves almost nothing; collapsing 15 images into 1–2 is a ~10× visual-token
  cut and frame-to-frame comparison within one image is something vision
  encoders handle well.
- Second lever: fewer frames. Thinking stays off (see Inference-call
  discipline). Frame-count handling is length-driven (`len(frames)`) — Stream
  1 produces *up to* 15 frames, not always exactly 15.

## Testing

- **Unit (no model):** `schema.py` accept/reject cases incl. JSON embedded in
  prose and thinking blocks, last-vs-first JSON selection, nullable run-1
  input fields; `prompts.py` message structure (telemetry present with both
  timeline annotations, frames labeled by filename mapping, staged-protocol +
  dual-citation instructions present, no bug-specific words like "drop" in
  the assembled **system prompt + ask template**); `images.py` encoding,
  event-frame pinning under subsampling, contact-sheet tiling.
- **Contract:** `analyze_run` with `mock_client` against the real
  `stream1_simulation/outputs/run1..3` fixtures — consuming exactly what
  `check_gemma_contract.py` produces, returning the README shape, across a
  chained run1→run2→run3 sequence (cumulative state), including the 3-arg
  degraded path (`next_to_check` forced null).
- **Repair path:** mock client returns malformed JSON first, valid JSON on
  retry (repair request must contain no images); malformed twice →
  `AnalyzeRunError(kind="bad_json")`. Timeout and backend-down →
  respective error kinds.
- **Live smoke (manual, model running):** `python -m
  stream2_gemma_inference.smoke --run_id 1` — the hour-0 two-image vision
  gate, then one real `analyze_run` call; prints latency, `prompt_tokens`,
  and the validated JSON. Not in CI.

## Risks / open items

- **Known data leak (decision: accept, reword the claim).** The manifest
  field `drop_step` and the chart's burned-in "object dropped, step N"
  annotation (`record_utils.py`) hand the model the *event* by name. Per the
  2026-07-04 discussion: facts belong to the model — what the demo
  demonstrates is the causal chain from un-punished event to missing reward
  term to concrete reward edit, which no input spells out. If the team wants
  a stronger "it discovered it" story, ask Stream 1 for an unannotated
  Gemma-facing chart copy and rename `drop_step` in the prompt to
  `episode_end_event_step`; not required for the demo to be honest about
  what it shows.
- **Cross-stream flags for Stream 3/4's owners:** (1) pass `manifest` as the
  4th argument; (2) returned `confirmed`/`ruled_out` are cumulative state,
  persist verbatim; (3) citation-in-string convention, if the UI wants to
  style it.
- **Gemma JSON discipline** — mitigated by low temperature, schema-in-prompt,
  last-fenced-block extraction, one repair round; hours 12–18 remain reserved
  for prompt hardening against real outputs. If discipline is still poor,
  the escape hatch is grammar-constrained `response_format`/GBNF on the
  final call of a two-call split (analyze in prose → formalize to JSON) —
  noted, not default, since a grammar from token 0 forbids the staged prose.
- **Latency unknown** until the model runs locally; measured hour ~6, levers
  specified above.
- **Model naming**: build plan says "Gemma 4 12B"; use whatever pre-quantized
  12B vision GGUF + projector the chosen server actually loads — client
  takes the model name from config, and the hour-0 vision gate validates it.
