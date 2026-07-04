# 24-Hour Build Plan

## The Full Pipeline

```
Simulation (PyBullet)
    → Rollout video + reward curve image
        → Gemma 12B (local) analyzes both
            → Hypothesis generated + stored in Antigravity
                → Computer Use pulls specific timesteps from WandB
                    → Updated hypothesis surfaced on demo UI
                        → Repeat across 3 runs
```

## Stream 1 — Simulation & Data Generation
*Owner: Robotics*

- **0–4h:** PyBullet pushing task. Introduce an obvious reward bug (e.g. missing
  penalty for dropping the object). Get a policy training and checkpointing.
- **4–12h:** Rollout recorder — sequential frames + reward curve PNG. Clean,
  consistent format; this is exactly what Gemma consumes.
- **12–18h:** Run 3 training sessions on the buggy reward. Label Run 1/2/3.
  These are your live demo assets.
- **18–24h:** Support integration, help Stream 4 with the demo script, verify
  outputs render correctly on screen.

**Deliverable:** 3 sets of rollout videos + reward curve images.

## Stream 2 — Gemma 12B Inference Pipeline
*Owner: RL*

- **0–4h:** Gemma 4 12B running locally via llama.cpp or LM Studio. Use a
  pre-quantized GGUF so it fits in 16GB VRAM.
- **4–12h:** `analyze_run(frames, chart, hypothesis_log) → updated_hypothesis_json`.
  Enable thinking mode via `<|think|>` in the system prompt. Test with dummy
  inputs first.
- **12–18h:** Lock down the system prompt so output JSON is consistent every
  time — confirmed failures, ruled-out failures, proposed reward edit, what to
  check next. Most important thing to get right.
- **18–24h:** Optimize for speed — fewer frames, reduce visual token budget
  (560 → 280) if needed. Target: under 60s per inference call.

**Deliverable:** `analyze_run(frames, chart, hypothesis_log) → updated_hypothesis_json`

## Stream 3 — Antigravity State + Agent Loop
*Owner: Platform*

- **0–4h:** Set up Antigravity, or a local JSON-file fallback if API access
  isn't confirmed. Each entry: run number, observed, confirmed, ruled out,
  next to check.
- **4–12h:** Agent loop:
  ```
  for each run:
      load hypothesis_log from Antigravity
      call Gemma inference (Stream 2)
      update hypothesis_log
      save back to Antigravity
      call Computer Use trigger (Stream 4)
      surface result to UI (Stream 4)
  ```
- **12–18h:** Wire in Stream 2's real inference function. Test full 3-run
  cycle end to end, dummy outputs first, then real.
- **18–24h:** Make the hypothesis log human-readable for the demo — hypotheses
  visibly added, confirmed, or crossed out per run.

**Deliverable:** Working agent loop across 3 sessions, persistent state, final diagnosis.

## Stream 4 — Computer Use + Demo UI
*Owner: Frontend/Backend*

- **0–4h:** WandB logging inside Stream 1's sim, reward curves auto-logged.
  Dummy workspace so Computer Use has something to navigate early.
- **4–12h:** Computer Use integration — given a timestep range, drive a
  headless browser (Playwright) to that exact WandB view, screenshot it,
  return it.
- **12–18h:** Gradio demo UI — 3 panels: rollout video, reward curve with
  flagged timestep highlighted, live hypothesis log.
- **18–24h:** 3 full dry runs, timed (target under 8 min). Fix what breaks.
  Prepare one-sentence explanation per panel.

**Deliverable:** Working demo UI + Computer Use integration + rehearsed script.

## Integration Checkpoints

- **Hour 4:** Data contract locked (hypothesis log shape, Computer Use trigger shape).
- **Hour 12:** Each stream has a working component in isolation, tested against mocks.
- **Hour 18:** First full end-to-end run, even if rough.
- **Hour 24:** Rehearsed demo.

## Known Risk

Statement Four ("Computer Use as load-bearing primitive") is arguably not
strictly true here — Gemma + persistent hypothesis state does the real work;
Computer Use is a screenshot-fetcher for one late-pipeline step. Worth having
a ready answer for judges: it's the mechanism that lets the agent *look at
exactly what it asked to look at* rather than a human manually scrubbing
WandB, which is itself part of the "automate the debugging loop" pitch.
