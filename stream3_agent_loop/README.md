# Stream 3 — Antigravity State + Agent Loop

Owns persistent hypothesis state and the orchestration loop tying Streams
1, 2, and 4 together. See root `docs/BUILD_PLAN.md` for the hour-by-hour
plan.

**Deliverable:** a working loop across 3 runs:

```
for each run:
    load hypothesis_log from Antigravity
    call Gemma inference (Stream 2: analyze_run)
    update hypothesis_log
    save back to Antigravity
    call Computer Use trigger (Stream 4: fetch_wandb_screenshot)
    surface result to UI (Stream 4: app.py)
```

Local fallback if Antigravity API access isn't confirmed: persist
`hypothesis_log` as JSON matching the shape in root README's "Data Contract"
section — that's the exact shape Stream 4's mocks already use.
