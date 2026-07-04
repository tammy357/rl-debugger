# Stream 4 — Computer Use + Demo UI

Owns everything the audience actually sees, plus the WandB screenshot
integration.

## Files

- `computer_use.py` — Playwright-based WandB screenshot fetcher. Falls back
  to a mock screenshot if `WANDB_RUN_URL` isn't set, so this works standalone.
- `app.py` — Gradio 3-panel demo UI (rollout / reward curve / hypothesis log).
- `mock_data/` — placeholder frames, reward curve, and hypothesis log so the
  UI can be built before Stream 1/2/3 have real outputs.

## Quickstart (standalone, no other streams needed yet)

```bash
pip install -r requirements.txt
playwright install chromium

python mock_data/generate_placeholders.py   # creates sample frames + reward_curve.png
python app.py                                # launches Gradio at localhost
```

## Swapping in real data

- Replace `mock_data/sample_frames/*.png` with Stream 1's real rollout frames.
- Replace `mock_data/reward_curve.png` with Stream 1's real chart export.
- Replace `mock_data/run_N_hypothesis.json` with Stream 3's real Antigravity
  output (same JSON shape — see root README's "Data Contract").
- Set `WANDB_RUN_URL` once Stream 1's logging is live, so `computer_use.py`
  hits the real workspace instead of mock mode.

## Open TODOs

- [ ] Confirm exact WandB chart selectors once a real run exists (chart
  scrubbing via UI is fragile — consider the WandB API/export instead of
  pixel-perfect screenshotting if judges don't need to see the literal
  WandB UI).
- [ ] Wire `app.py` to poll Stream 3's agent loop live instead of reading
  static JSON, once that loop exists.
- [ ] Rehearse full demo timing (target: under 8 minutes).
