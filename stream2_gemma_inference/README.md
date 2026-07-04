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
llama-server -hf <team's gemma-12b vision GGUF, e.g. ggml-org/gemma-3-12b-it-GGUF> \
             -c 16384 --port 8080
# then ALWAYS run the vision gate before building on it:
python -m stream2_gemma_inference.smoke --run_id 1 --gate-only
```

Context ≥ 8k is required (15 frames + chart ≈ 4.5k visual tokens). Config via
env: `GEMMA_BASE_URL` (default `http://localhost:8080/v1`), `GEMMA_MODEL`,
`GEMMA_N_FRAMES` (default 8; the drop/event frame always survives
subsampling), `GEMMA_CONTACT_SHEET=1` (first latency lever).

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
