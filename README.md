<div align="center">

# 🔧 RL Policy Debugger

**An agent that watches your robot's failed training runs and tells you *why* — entirely offline.**

Built for the Google DeepMind Track · Statement Five (Remote)
*"Best mobile, web, or edge application running Gemma locally for offline, privacy-first inference."*

[![Gemma](https://img.shields.io/badge/Gemma-4%2012B-4285F4?logo=google)](https://ai.google.dev/gemma)
[![Local Only](https://img.shields.io/badge/inference-100%25%20local-2ecc71)](https://claude.ai/chat/134db251-11ee-47dc-a492-471f85706a0e)
[![PyBullet](https://img.shields.io/badge/sim-PyBullet-orange)](https://claude.ai/chat/134db251-11ee-47dc-a492-471f85706a0e)
[![Gradio](https://img.shields.io/badge/UI-Gradio-fb923c)](https://claude.ai/chat/134db251-11ee-47dc-a492-471f85706a0e)

</div>

---

## 💡 The idea

Debugging a broken reward function usually means a human staring at rollout
videos next to reward curves, run after run, trying to spot the pattern.
**RL Policy Debugger automates that loop.** It watches each training run's
rollout + reward curve + logs, forms a hypothesis about what's going wrong,
and *remembers* across runs — confirming or ruling out ideas as new evidence
comes in, the same way a human researcher builds understanding over time.

Everything runs on **Gemma 4 12B, entirely on your own machine.** No cloud
calls, no uploaded training data — works in an air-gapped robotics lab.

> Not a dashboard. Not an image classifier. An agent that reasons across
> three runs and lands on a concrete, proposed reward fix.

## 🎥 Demo

> *Demo video link goes here*

Walkthrough: buggy rollout → Gemma reasoning about the rollout + logs →
hypothesis log evolving across runs 1 → 2 → 3 → a concrete proposed reward
edit.

## 🧠 How it works

```
 Simulation (PyBullet)
        │
        ▼
 Rollout video + reward curve + run logs
        │
        ▼
 Gemma 4 12B  (local, multimodal)  ── reasons over video + chart + logs together
        │
        ▼
 Hypothesis: confirmed / ruled out / next to check / proposed reward fix
        │
        ▼
 Persisted locally, carried into the next run
        │
        ▼
 Surfaced live on the demo UI  ──  repeat across 3 runs
```

## ✨ What makes this interesting

- **Multimodal reasoning, not just vision.** Gemma reads the rollout video
and the reward curve and the run's structured logs together in one prompt —
the same inputs a human RL researcher would use.
- **Persistent memory across runs.** Hypotheses aren't regenerated from
scratch each time — they're confirmed, ruled out, and refined as more
evidence comes in across three separate training runs.
- **Fully offline.** No API keys, no cloud inference, no training data ever
leaves the machine. Built to work in an air-gapped setting.

## 🗂️ Project structure

```
rl-policy-debugger/
├── stream1_simulation/       # PyBullet task, deliberately buggy reward, rollout export
├── stream2_gemma_inference/  # local Gemma 4 12B multimodal inference
├── stream3_agent_loop/       # persistent hypothesis state + orchestration loop
├── stream4_demo_ui/          # Gradio demo UI
└── CONTRIBUTING.md           # data contracts, setup details, internal notes
```

## 🚀 Quickstart

```
python -m venv venv && source venv/bin/activate

pip install -r stream1_simulation/requirements.txt
pip install -r stream2_gemma_inference/requirements.txt
pip install -r stream3_agent_loop/requirements.txt
pip install -r stream4_demo_ui/requirements.txt
```

Run Gemma locally:

```
brew install llama.cpp
llama-server -hf ggml-org/gemma-4-12B-it-GGUF -c 8192 --port 8080 --reasoning-budget 0
```

Run the agent loop, then launch the demo:

```
python stream3_agent_loop/agent_loop.py --run 1 --run 2 --run 3
python stream4_demo_ui/app.py
```

Full setup details, the data contract between streams, and internal
engineering notes: see CONTRIBUTING.md.

## 🛠️ Built with

PyBullet · Stable-Baselines3 · Gemma 4 12B (llama.cpp) · Gradio · Python

## 👥 Team

Stream
Owner

Simulation & Data Generation
Robotics

Gemma 12B Inference
RL

Agent Loop & Persistent State
Platform

Demo UI
Frontend/Backend

---

<div align="center">
<sub>Built in 24 hours for the Google DeepMind track · Statement Five (Remote)</sub>
</div>
