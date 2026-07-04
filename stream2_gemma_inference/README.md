# Stream 2 — Gemma 12B Inference Pipeline

*Owner: RL*

**Deliverable (live):**

```python
from stream2_gemma_inference import analyze_run

entry = analyze_run(frames, chart, hypothesis_log, manifest)  # manifest: pass it!
```

Design: `docs/superpowers/specs/2026-07-04-analyze-run-design.md`.

## Server (start here, hour 0)

Primary backend is llama.cpp's `llama-server` (prebuilt, reliable Gemma
vision); LM Studio is the fallback if it loads the model's vision path:

```bash
brew install llama.cpp
llama-server -hf ggml-org/gemma-4-12B-it-GGUF \
             -c 8192 --port 8080 --reasoning-budget 0
# then ALWAYS run the vision gate before building on it:
python -m stream2_gemma_inference.smoke --run_id 1 --gate-only
```

**`--reasoning-budget 0` is mandatory**: Gemma 4's chat template enables
thinking by default, and the model will spend its *entire* `max_tokens` budget
on `reasoning_content`, returning zero answer tokens (found live 2026-07-04;
the client now raises an actionable error if this happens). If `-hf`'s
built-in downloader flakes (status code -1), fetch with
`hf download ggml-org/gemma-4-12B-it-GGUF gemma-4-12B-it-Q4_K_M.gguf
mmproj-gemma-4-12B-it-Q8_0.gguf` and pass `-m`/`--mmproj` directly.

Context ≥ 8k is required. Config via env: `GEMMA_BASE_URL` (default
`http://localhost:8080/v1`), `GEMMA_MODEL`, `GEMMA_N_FRAMES` (default 8; the
drop/event frame always survives subsampling), `GEMMA_CONTACT_SHEET=1`
(latency lever). **Measured on an M1 Pro 16GB (Q4_K_M, 8 frames + chart):
32.9s per call, ~1.6k prompt tokens** — inside the <60s demo budget.

## For Stream 3 (cross-stream contract)

- Pass `manifest` (the run's `manifest.json` dict) as the 4th argument — you
  already load it for the Computer Use trigger. Without it,
  `next_to_check` is always `null` and `run` is stamped from the incoming log.
- Input `hypothesis_log` = the previous run's full entry (run 1: the empty
  bootstrap). The RETURN is the complete cumulative state — persist it
  verbatim, no merging.
- On failure, `AnalyzeRunError` with `.kind` ∈ `timeout | backend | bad_json`
  (`.raw_response` attached). Malformed data is never returned.
- Claim strings embed their evidence citations, e.g.
  `"no penalty at failure event (frames 12-13; drop_step=65, no dip)"`.

## Tests

```bash
./.venv/bin/python -m pytest stream2_gemma_inference/tests -v   # no model needed
python -m stream2_gemma_inference.smoke --run_id 1              # live, model needed
```
