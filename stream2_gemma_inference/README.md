# Stream 2 — Gemma 12B Inference Pipeline

*Owner: RL*

Owns local Gemma 4 12B inference. See root `docs/BUILD_PLAN.md` for the
hour-by-hour plan.

**Deliverable:** a single callable function

```python
def analyze_run(frames: list[str], chart: str, hypothesis_log: dict) -> dict:
    """Returns an updated hypothesis JSON matching the shape in root README's
    'Data Contract' section."""
```

Stream 3 imports this directly into the agent loop.
