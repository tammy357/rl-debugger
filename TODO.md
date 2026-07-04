## 24-Hour RL Debugger Build Plan

---

### The Full Pipeline (What You're Building)

```
Simulation (PyBullet)
    → Rollout video + reward curve image
        → Gemma 12B (local) analyzes both
            → Hypothesis generated + stored in Antigravity
                → Computer Use pulls specific timesteps from WandB
                    → Updated hypothesis surfaced on demo UI
                        → Repeat across 3 runs
```

---

### 4 Parallel Work Streams

---

**Stream 1 — Simulation & Data Generation**
*Owner: Robotics member*

Owns everything that produces the data the agent reasons about.

Hours 0–4: Set up PyBullet with a simple pushing task. Deliberately introduce a reward bug — something obvious like missing a penalty for dropping the object. Get a policy training and saving checkpoints.

Hours 4–12: Write a rollout recorder that saves video clips (as sequential frames) and exports reward curves as chart images. These are the exact inputs Gemma will receive. Make sure output format is clean — numbered frames, consistent image size, reward curve saved as a PNG.

Hours 12–18: Run 3 training sessions with the buggy reward, saving rollout videos and charts for each. Label them Run 1, Run 2, Run 3. These become your live demo assets.

Hours 18–24: Support integration, help Stream 4 with the demo script, verify the simulation outputs look right on screen.

**Deliverable:** Three sets of rollout videos + reward curve images, cleanly formatted and ready to feed into Gemma.

---

**Stream 2 — Gemma 12B Inference Pipeline**
*Owner: ML member*

Owns everything that makes Gemma reason correctly about the inputs.

Hours 0–4: Get Gemma 4 12B running locally via llama.cpp or LM Studio. Verify it loads, verify basic inference works. Don't overthink the setup — use the pre-quantized GGUF from Hugging Face so it fits in 16GB VRAM.

Hours 4–12: Build the inference function. It takes a list of video frames + a reward curve image + the current hypothesis log as inputs, and outputs a structured JSON hypothesis update. Enable thinking mode with `<|think|>` in the system prompt. Test this in isolation with dummy inputs before touching anything else.

Hours 12–18: Refine the system prompt so Gemma outputs consistent structured JSON every time — failure modes confirmed, failure modes ruled out, proposed reward edit, what to look at next. This is the most important thing to get right because everything downstream depends on it.

Hours 18–24: Optimize — if inference is slow, reduce the number of frames passed per run, adjust the visual token budget down from 560 to 280. Make sure a full inference call completes in under 60 seconds for the demo.

**Deliverable:** A single callable function `analyze_run(frames, chart, hypothesis_log) → updated_hypothesis_json`

---

**Stream 3 — Antigravity State + Agent Loop**
*Owner: Physical AI member*

Owns the memory system and the multi-step agent logic that ties everything together.

Hours 0–4: Set up Antigravity. If API access isn't confirmed, build a local fallback — a simple JSON file that persists the hypothesis log between runs. The structure matters: each entry should have the run number, what was observed, what was confirmed, what was ruled out, and what to check next.

Hours 4–12: Build the agent loop. This is the core orchestration logic:
```
for each run:
    load hypothesis_log from Antigravity
    call Gemma inference (Stream 2)
    update hypothesis_log
    save back to Antigravity
    call Computer Use trigger (Stream 4)
    surface result to UI (Stream 4)
```

Hours 12–18: Wire Stream 2's inference function into this loop. Test the full three-run cycle end to end with dummy Gemma outputs first, then with real ones.

Hours 18–24: Make the hypothesis log human-readable for the demo screen. Each run should visibly show hypotheses being added, confirmed, or crossed out. This is what the audience watches.

**Deliverable:** A working agent loop that runs three sessions, updates persistent state, and produces a final diagnosis.

---

**Stream 4 — Computer Use + Demo UI**
*Owner: Fourth member*

Owns everything the audience actually sees.

Hours 0–4: Set up WandB logging inside Stream 1's simulation so reward curves are automatically logged. Get a basic WandB workspace running with dummy data so Computer Use has something to navigate.

Hours 4–12: Build the Computer Use integration. The agent receives a timestep range from Antigravity (e.g., "check steps 47,000–52,000 in Run 2") and Computer Use navigates to that exact view in WandB, takes a screenshot, and returns it. This doesn't need to be fancy — headless browser automation with Playwright is enough.

Hours 12–18: Build the demo UI in Gradio. Three panels: left shows the current rollout video clip, center shows the reward curve with the flagged timestep highlighted, right shows the live hypothesis log updating in real time. Keep it simple — judges look at this for 5 minutes.

Hours 18–24: Full dry run of the demo three times. Time it. It should be completable in under 8 minutes. Fix anything that breaks during the dry run. Prepare the one-sentence explanation of what each screen shows.

**Deliverable:** A working demo UI and Computer Use integration, plus a rehearsed demo script.

---

### Integration Checkpoints

| Time | Checkpoint |
|---|---|
| Hour 6 | Stream 1 has first rollout video. Stream 2 has Gemma running locally. Quick sync — verify frames from Stream 1 can be passed to Stream 2 |
| Hour 12 | Stream 2's inference function works end to end. Stream 3's agent loop works with dummy data. Streams 1+2+3 do a first full integration test |
| Hour 18 | Full three-run cycle works. Stream 4 wires UI around it. First full demo dry run |
| Hour 22 | Final dry run. Fix only critical bugs. Everyone stops adding features |
| Hour 24 | Submit |

---

### What To Cut If You're Running Behind

If hour 18 arrives and things aren't integrated, cut in this order:

- Cut Computer Use first — replace it with a hardcoded timestep highlight on the reward curve chart. The agent still works, it just doesn't navigate WandB live
- Cut the Gradio UI — run the demo from a Jupyter notebook instead, it's less pretty but completely functional
- Never cut Antigravity — that's the track requirement
- Never cut Gemma's multimodal reasoning — that's the core of the project