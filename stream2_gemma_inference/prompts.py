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
