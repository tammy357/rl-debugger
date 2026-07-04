# RL Policy Debugger

An RL debugging agent that watches your robot-training rollouts and reward
curves, and diagnoses reward bugs — **entirely on your own machine**. Uses
**Gemma 4 12B** (local) as the reasoning core, with native video/image
understanding to analyze rollouts and reward charts in a single unified
prompt. Training data and policy behavior never leave your laptop; works in
an air-gapped robotics lab.

Built in 24 hours across 4 parallel streams for the Google DeepMind track,
**Statement Five (Remote): "Best mobile, web, or edge application running
Gemma locally for offline, privacy-first inference."**

The differentiator: the agent builds and refines failure hypotheses
*persistently across training runs* — hypotheses added, confirmed, or
crossed out from run 1 → 2 → 3, ending in a concrete proposed reward fix.
Not a dashboard, not an image analyzer — an agent that reasons across runs
the way a human RL researcher does.

---

## Pipeline

```
Simulation (PyBullet)
    → Rollout video + reward curve
        → Gemma 12B (local) analyzes both, alongside the run's log data
            → Hypothesis generated + persisted (local JSON state)
                → Flagged training-step range highlighted on the reward curve
                    → Updated hypothesis surfaced on demo UI
                        → Repeat across 3 runs
```

## Repo Structure

```
rl-policy-debugger/
├── stream1_simulation/       # PyBullet env, reward bug, rollout recorder
├── stream2_gemma_inference/  # local Gemma 12B inference pipeline
├── stream3_agent_loop/       # local JSON state + orchestration loop
├── stream4_demo_ui/          # reward-curve highlighting + Gradio UI
├── docs/                     # shared contracts, integration notes
└── README.md
```

Each `streamN_*/` folder has its own `requirements.txt` — install only the
one for your lane, so you're never fighting a teammate's build (e.g.
pybullet's C++ compile) just to run your own code.

## Ownership

| Stream | Owns | Member |
|---|---|---|
| 1 — Simulation & Data Generation | PyBullet task, reward bug, rollout + reward-curve exports | Robotics |
| 2 — Gemma 12B Inference | `analyze_run(frames, chart, hypothesis_log, manifest) → updated_hypothesis_json` | RL |
| 3 — Local State + Agent Loop | Persistent hypothesis state (local JSON — this *is* the offline story, not a fallback), orchestration loop | Platform |
| 4 — Demo UI | Local reward-curve highlighting, Gradio demo | Frontend/Backend |

Full hour-by-hour plan for each stream: see `docs/BUILD_PLAN.md` (note: the
plan's Computer Use/WandB sections and "never cut Antigravity" note are
stale — see Track Correction below).

## Track Correction (read this first)

We're a **remote** team, which puts us in **Statement Five**, not Four —
Four (Computer Use / Antigravity / Live Translate) is **in-person only**.
This is good news: it cuts work and fits our project better.

- **Stream 1 (sim):** no change — already fully local.
- **Stream 2 (Gemma):** no change.
- **Stream 3 (agent loop):** local JSON persistence is the *preferred* path
  now, not a fallback — cloud state (Antigravity) would contradict the
  offline/privacy-first pitch. The 3-run persistent hypothesis memory stays;
  that's the creativity differentiator.
- **Stream 4 (demo):** Computer Use / WandB navigation is removed — it was
  Statement Four's requirement and undercuts the offline story. Replaced
  with rendering the training reward curve locally and highlighting the
  agent's `next_to_check.step_range` directly on it.

## The Data Contract (agree on this first)

Everything downstream of Stream 1/2/3 depends on this shape being fixed early.

**`analyze_run(frames, chart, hypothesis_log, manifest)`** — note the order:
hypothesis_log third, manifest fourth. **Call with keywords, not positionally**
— `analyze_run(frames, chart, hypothesis_log=log, manifest=manifest)` — since
swapping the two dicts positionally fails silently with no error and garbage
output. `manifest` must always be passed; without it, `next_to_check` is
forced to null by design. `manifest` is Stream 1's `manifest.json` for that
run as-is (see `stream1_simulation/check_gemma_contract.py` for the exact
shape: `run`, `checkpoint_step`, `step_range`, `drop_step`, `total_reward`,
`num_episode_steps`, `success`, `frames`, `reward_curve`).

Gemma reasons over the rollout video/chart *and* the run's log data together
— the same way a human RL researcher reads a rollout next to its logs rather
than judging from pixels alone.

**Return value** is the complete cumulative hypothesis state, **not a
delta** — persist it verbatim. Merging would make run-1 findings vanish from
the UI on run 2.

**Errors:** `AnalyzeRunError` with `.kind ∈ {timeout, backend, bad_json}` —
malformed data never comes back as a normal return, so callers branch on
`.kind` instead of parsing garbage.

**Hypothesis log entry:**
```json
{
  "run": 2,
  "timestamp": "2026-01-01T12:00:00Z",
  "confirmed": ["missing penalty for dropping object"],
  "ruled_out": ["gripper friction too low"],
  "next_to_check": {
    "run": 2,
    "step_range": [47000, 52000],
    "reason": "reward spikes right before object drop"
  },
  "proposed_reward_edit": "add -1.0 penalty on object.z < table_height"
}
```
Note: `next_to_check` can legitimately be `null` after a run where the agent
is confident/converged — callers must handle that case, not treat it as an
error. Also note `step_range` here is in **training-step units** (the full
training run, e.g. 0–80000), which is a different axis than the per-episode
rollout curves in `stream1_simulation/outputs/run{N}/reward_curve.png`
(0–~65 rollout steps) — the two aren't interchangeable.

## Setup

Install only your own stream's dependencies — no need to fight a teammate's
build just to run your own code:

```bash
python -m venv venv && source venv/bin/activate
# or: conda create -n rl-debugger python=3.10 -y && conda activate rl-debugger

pip install -r stream1_simulation/requirements.txt       # robotics
pip install -r stream2_gemma_inference/requirements.txt  # RL
pip install -r stream3_agent_loop/requirements.txt       # platform
pip install -r stream4_demo_ui/requirements.txt          # frontend/backend
```

Running Gemma locally (Stream 2):
```bash
brew install llama.cpp
llama-server -hf ggml-org/gemma-4-12B-it-GGUF -c 8192 --port 8080 --reasoning-budget 0
```
`--reasoning-budget 0` is mandatory — Gemma 4 thinks by default and will
burn the whole token budget on hidden reasoning, returning an empty answer.
~30s per call measured on an M1 Pro 16GB, Q4_K_M.

## Demo

Three-panel Gradio UI: rollout video (left), reward curve with the agent's
flagged training-step range highlighted (center), live hypothesis log
(right). Target: full walkthrough under 8 minutes.

**Pitch:** lead with the agent reasoning across 3 runs (hypotheses added →
confirmed → crossed out → concrete reward fix). Say "running locally / fully
offline" early and out loud. Avoid framing this as "a dashboard" or "an
image analyzer" in the pitch.

## Submission Checklist (due Sunday 12:00 CEST)

- [ ] Repo set to **public**
- [ ] 1-minute demo video (YouTube/Loom) — Demo is 50% of the score. Show:
      buggy rollout → Gemma reasoning → hypothesis log evolving runs 1→2→3 →
      proposed reward fix.
- [ ] Project description makes clear everything was built during the event
- [ ] Submit at cerebralvalley.ai/e/raise-summit-hackathon

Judging weights: Demo 50 · Impact 25 · Creativity 15 · Pitch 10

## Status

See each `streamN_*/README.md` for stream-specific setup and current state.
