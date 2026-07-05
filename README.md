<div align="center">

# 🔧 RL Policy Debugger

### An agent that watches your robot's failed training runs and tells you *why* — entirely offline.

**Google DeepMind Track · Statement Five (Remote) — The Edge / On-Device Track**
*"Best mobile, web, or edge application running Gemma locally for offline, privacy-first inference."*

[![Gemma](https://img.shields.io/badge/Gemma-4%2012B-4285F4?logo=google&logoColor=white)](https://ai.google.dev/gemma)
[![100% Local](https://img.shields.io/badge/inference-100%25%20local-2ecc71)]()
[![Zero Cloud Calls](https://img.shields.io/badge/cloud%20calls-zero-2ecc71)]()
[![PyBullet](https://img.shields.io/badge/sim-PyBullet-orange)]()
[![Gradio](https://img.shields.io/badge/UI-Gradio-fb923c)]()
[![Built in 24h](https://img.shields.io/badge/built%20in-24%20hours-lightgrey)]()

**[🎥 Watch the demo](#-demo)** · **[🚀 Quickstart](#-quickstart)** · **[🧠 How it works](#-how-it-works)**

</div>

---

## The problem

Debugging a broken reward function usually means an RL researcher staring at
rollout videos next to reward curves, run after run, trying to spot the
pattern by hand. It's slow, it's manual, and it doesn't scale past one
engineer's attention span.

## The solution

**RL Policy Debugger automates that loop.** It watches each training run's
rollout video, reward curve, and structured logs together, forms a
hypothesis about what's going wrong, and *remembers* across runs —
confirming or ruling out ideas as new evidence comes in, the same way a
human researcher builds understanding over days of work.

Everything runs on **Gemma 4 12B, entirely on your own machine.** No cloud
calls, no uploaded training data, no API keys — it works in an air-gapped
robotics lab, which is exactly where most real RL debugging happens.

> Not a dashboard. Not an image classifier. An agent that reasons across
ten independent training runs — across multiple, different reward bugs —
and lands on a concrete, proposed reward fix.

## Who this is for

Any team training robot policies who has ever lost an afternoon to "why did
this run fail?" — from a solo robotics researcher on a laptop to a lab that
can't send proprietary training data off-device.

## 🎥 Demo

> **[▶ Watch the 1-minute demo](#)** — replace with your video link before submitting

Walkthrough: a buggy rollout → Gemma reasoning about the rollout + reward
curve + logs together → the hypothesis log evolving as it's tested across
multiple runs → a concrete, proposed reward-function fix.

## 🧠 How it works

```
 Simulation (PyBullet)
        │
        ▼
 Rollout video + reward curve + structured run logs
        │
        ▼
 Gemma 4 12B  (local, multimodal)
        │   reasons over video + chart + logs together —
        │   the same evidence a human RL researcher would use
        ▼
 Hypothesis:  confirmed  ·  ruled out  ·  next to check  ·  proposed reward fix
        │
        ▼
 Persisted locally — carried forward into the next run, not regenerated from scratch
        │
        ▼
 Surfaced live on the demo UI  ──  repeated across 10 independent training runs
```

## ✨ What makes this interesting

- **Multimodal reasoning, not just vision.** Gemma reads the rollout video,
  the reward curve, *and* the run's structured logs (drop step, total
  reward, success/fail) together in a single prompt — not just pattern
  matching on pixels.
- **Persistent memory across runs.** Hypotheses are confirmed, ruled out,
  and refined as more evidence arrives — the agent isn't re-analyzing from
  scratch each time, it's building a case across runs.
- **Proven to generalize.** Tested across ten independent training runs,
  including multiple distinct reward bugs — not tuned to recognize one
  specific failure.
- **Fully offline, by design, not by accident.** No API keys, no cloud
  inference, no training data ever leaves the machine. Built for the exact
  setting real RL debugging happens in: an air-gapped robotics lab.

## 🗂️ Project structure

```
rl-policy-debugger/
├── stream1_simulation/       # PyBullet task, deliberately buggy rewards, rollout export
├── stream2_gemma_inference/  # local Gemma 4 12B multimodal inference
├── stream3_agent_loop/       # persistent hypothesis state + orchestration loop
├── stream4_demo_ui/          # Gradio demo UI
└── CONTRIBUTING.md           # data contracts, setup details, internal engineering notes
```

## 🚀 Quickstart

```bash
python -m venv venv && source venv/bin/activate

pip install -r stream1_simulation/requirements.txt
pip install -r stream2_gemma_inference/requirements.txt
pip install -r stream3_agent_loop/requirements.txt
pip install -r stream4_demo_ui/requirements.txt
```

Run Gemma locally:
```bash
brew install llama.cpp
llama-server -hf ggml-org/gemma-4-12B-it-GGUF -c 8192 --port 8080 --reasoning-budget 0
```
> `--reasoning-budget 0` is required — Gemma 4 thinks by default and will
> burn its whole token budget on hidden reasoning otherwise. ~30s per call
> on an M1 Pro 16GB, Q4_K_M.

Run the agent loop across all training runs, then launch the demo:
```bash
python stream3_agent_loop/agent_loop.py --run 1 --run 2 --run 3 --run 4 --run 5 --run 6 --run 7 --run 8 --run 9 --run 10
python stream4_demo_ui/app.py
```

Full setup details, the data contract between streams, and internal
engineering notes: see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## 🛠️ Built with

PyBullet · Stable-Baselines3 · Gemma 4 12B (llama.cpp) · Gradio · Python

## 👥 Team

| Stream | Owns | Owner |
|---|---|---|
| Simulation & Data Generation | PyBullet task, reward bugs, rollout exports | Robotics |
| Gemma 12B Inference | Local multimodal reasoning core | RL |
| Agent Loop & Persistent State | Cross-run hypothesis memory, orchestration | Platform |
| Demo UI | Live Gradio walkthrough | Frontend/Backend |

## 📄 License

MIT — see [`LICENSE`](./LICENSE).

---

<div align="center">
<sub>Built in 24 hours for the Google DeepMind Track · Statement Five (Remote) · RAISE Hackathon, Paris 2026</sub>
</div>
