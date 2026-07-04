"""
Stream 4 — Demo UI.

Three panels:
  left   - current rollout video/frame
  center - reward curve with flagged timestep highlighted
  right  - live hypothesis log (auto-refreshes, highlights new items)

Run standalone against mock data:
    python mock_data/generate_placeholders.py   # once, to create mock assets
    python app.py
"""
import glob
import json
import os

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(HERE, "mock_data")
STREAM1_OUTPUTS = os.path.join(HERE, "..", "stream1_simulation", "outputs")

AUTO_REFRESH_SECONDS = 4  # how often the demo polls for new hypothesis data


def load_manifest(run: int) -> dict | None:
    """Stream 1's real output: stream1_simulation/outputs/run{N}/manifest.json"""
    path = os.path.join(STREAM1_OUTPUTS, f"run{run}", "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_hypothesis_data(run: int) -> dict | None:
    """Real hypothesis log will come from Stream 3 once it exists (see
    check_gemma_contract.py's dummy stand-in for the exact shape). Falls back
    to the Stream 4 mock JSON if Stream 3 hasn't produced one yet."""
    mock_path = os.path.join(MOCK_DIR, f"run_{run}_hypothesis.json")
    if os.path.exists(mock_path):
        with open(mock_path) as f:
            return json.load(f)
    return None


def format_hypothesis(data: dict, prev_confirmed: set, prev_ruled_out: set) -> str:
    """Renders the hypothesis log, marking items not seen on the previous
    poll with 🆕 so new findings visibly stand out during a live demo."""
    timestamp = data.get("timestamp", "")
    header = f"### Run {data['run']}" + (f" — {timestamp}" if timestamp else "")
    lines = [header, ""]

    lines.append("**Confirmed:**")
    for item in data.get("confirmed", []):
        tag = "🆕 " if item not in prev_confirmed else ""
        lines.append(f"- ✅ {tag}{item}")
    lines.append("")

    lines.append("**Ruled out:**")
    for item in data.get("ruled_out", []):
        tag = "🆕 " if item not in prev_ruled_out else ""
        lines.append(f"- ❌ {tag}{item}")
    lines.append("")

    nxt = data.get("next_to_check")
    if nxt:
        lines.append("**Next to check:**")
        lines.append(f"- Run {nxt.get('run')}, steps {nxt.get('step_range')}")
        lines.append(f"  _{nxt.get('reason', '')}_")

    edit = data.get("proposed_reward_edit")
    if edit:
        lines.append("")
        lines.append(f"**Proposed reward edit:** `{edit}`")

    return "\n".join(lines)


def load_frames(run: int) -> list[str]:
    """Prefers Stream 1's real frames (via manifest.json); falls back to
    Stream 4's mock frames if that run doesn't exist yet."""
    manifest = load_manifest(run)
    if manifest:
        out_dir = os.path.join(STREAM1_OUTPUTS, f"run{run}")
        frames = [os.path.join(out_dir, record["file"]) for record in manifest["frames"]]
        return sorted(frames)
    return sorted(glob.glob(os.path.join(MOCK_DIR, "sample_frames", "*.png")))


def load_reward_curve(run: int) -> str | None:
    """Prefers Stream 1's real reward_curve.png; falls back to the mock."""
    manifest = load_manifest(run)
    if manifest:
        out_dir = os.path.join(STREAM1_OUTPUTS, f"run{run}")
        path = os.path.join(out_dir, manifest["reward_curve"])
        if os.path.exists(path):
            return path
    mock_path = os.path.join(MOCK_DIR, "reward_curve.png")
    return mock_path if os.path.exists(mock_path) else None


def refresh(run: int, seen_state: dict):
    """seen_state tracks {"confirmed": [...], "ruled_out": [...]} from the
    last poll (lists, not sets — gr.State must stay JSON-serializable), so
    newly-appeared items can be flagged with 🆕."""
    seen_state = seen_state or {"confirmed": [], "ruled_out": []}
    prev_confirmed = set(seen_state["confirmed"])
    prev_ruled_out = set(seen_state["ruled_out"])

    frames = load_frames(run)
    curve = load_reward_curve(run)
    data = load_hypothesis_data(run)

    if data is None:
        return frames, curve, f"No hypothesis log found for run {run} yet.", seen_state

    hyp_md = format_hypothesis(data, prev_confirmed, prev_ruled_out)

    new_state = {
        "confirmed": list(data.get("confirmed", [])),
        "ruled_out": list(data.get("ruled_out", [])),
    }
    return frames, curve, hyp_md, new_state


with gr.Blocks(title="RL Policy Debugger") as demo:
    gr.Markdown("# RL Policy Debugger — Live Diagnosis")

    seen_items = gr.State({"confirmed": set(), "ruled_out": set()})

    with gr.Row():
        run_selector = gr.Number(value=2, label="Run", precision=0)
        refresh_btn = gr.Button("Refresh now")
        auto_refresh = gr.Checkbox(value=True, label="Auto-refresh")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Rollout")
            rollout_gallery = gr.Gallery(label="Rollout frames", columns=4, height=300)
        with gr.Column():
            gr.Markdown("### Reward Curve")
            reward_img = gr.Image(label="Reward curve (flagged timestep highlighted)")
        with gr.Column():
            gr.Markdown("### Hypothesis Log")
            hypothesis_md = gr.Markdown()

    outputs = [rollout_gallery, reward_img, hypothesis_md, seen_items]

    refresh_btn.click(fn=refresh, inputs=[run_selector, seen_items], outputs=outputs)
    demo.load(fn=refresh, inputs=[run_selector, seen_items], outputs=outputs)

    # Auto-refresh: gr.Timer polls on an interval and re-fires `refresh` while
    # the checkbox is on. This is what makes the hypothesis panel feel "live"
    # during the demo instead of requiring a manual click each time.
    timer = gr.Timer(AUTO_REFRESH_SECONDS, active=True)
    timer.tick(fn=refresh, inputs=[run_selector, seen_items], outputs=outputs)
    auto_refresh.change(fn=lambda on: gr.Timer(active=on), inputs=[auto_refresh], outputs=[timer])

if __name__ == "__main__":
    demo.launch()
