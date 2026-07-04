# RL Policy Debugger

An AI agent that watches robot training rollout videos and reward curves to
diagnose *why* a policy is failing — building and refining failure hypotheses
persistently across training sessions. Uses **Gemma 4 12B** (local) as the
reasoning core, with native video/image understanding to analyze rollouts and
reward charts in a single unified prompt.

Built in 24 hours across 4 parallel streams.

---

## Pipeline

```
Simulation (PyBullet)
    → Rollout video + reward curve image
        → Gemma 12B (local) analyzes both
            → Hypothesis generated + stored in Antigravity
                → Computer Use pulls specific timesteps from WandB
                    → Updated hypothesis surfaced on demo UI
                        → Repeat across 3 runs
```

## Repo Structure

```
rl-policy-debugger/
├── stream1_simulation/       # PyBullet env, reward bug, rollout recorder
├── stream2_gemma_inference/  # local Gemma 12B inference pipeline
├── stream3_agent_loop/       # Antigravity state + orchestration loop
├── stream4_demo_ui/          # Computer Use (Playwright/WandB) + Gradio UI
├── docs/                     # shared contracts, integration notes
├── requirements.txt
└── README.md
```

## Ownership

| Stream | Owns | Member |
|---|---|---|
| 1 — Simulation & Data Generation | PyBullet task, reward bug, rollout + reward-curve exports | Robotics |
| 2 — Gemma 12B Inference | `analyze_run(frames, chart, hypothesis_log) → updated_hypothesis_json` | RL |
| 3 — Antigravity + Agent Loop | Persistent hypothesis state, orchestration loop | Platform |
| 4 — Computer Use + Demo UI | WandB Computer Use integration, Gradio demo | Frontend/Backend |

Full hour-by-hour plan for each stream: see `docs/BUILD_PLAN.md`.

## The Data Contract (agree on this first)

Everything downstream of Stream 1/2/3 depends on this shape being fixed early.
Whoever finishes their piece first should build against these mocks, not wait.

**Hypothesis log entry** (produced by Stream 2, stored via Stream 3):
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

**Computer Use trigger** (Stream 3 → Stream 4):
```json
{ "run": 2, "step_range": [47000, 52000] }
```
Stream 4 returns a screenshot path/URL for that view.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # Stream 4 only
```

## Demo

Three-panel Gradio UI: rollout video (left), reward curve with flagged
timestep highlighted (center), live hypothesis log (right). Target: full
walkthrough under 8 minutes.

## Status

Early scaffold — see each `streamN_*/README.md` for stream-specific setup and
current state.
