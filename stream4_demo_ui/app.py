"""
Stream 4 — Demo UI.

Three panels:
  left   - current rollout video/frame
  center - reward curve with flagged timestep highlighted
  right  - live hypothesis log

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


def load_hypothesis_log(run: int) -> str:
    path = os.path.join(MOCK_DIR, f"run_{run}_hypothesis.json")
    if not os.path.exists(path):
        return f"No hypothesis log found for run {run} yet."
    with open(path) as f:
        data = json.load(f)
    return format_hypothesis(data)


def format_hypothesis(data: dict) -> str:
    lines = [f"### Run {data['run']} — {data['timestamp']}", ""]
    lines.append("**Confirmed:**")
    for item in data.get("confirmed", []):
        lines.append(f"- ✅ {item}")
    lines.append("")
    lines.append("**Ruled out:**")
    for item in data.get("ruled_out", []):
        lines.append(f"- ❌ {item}")
    lines.append("")
    nxt = data.get("next_to_check", {})
    if nxt:
        lines.append("**Next to check:**")
        lines.append(f"- Run {nxt.get('run')}, steps {nxt.get('step_range')}")
        lines.append(f"  _{nxt.get('reason', '')}_")
    edit = data.get("proposed_reward_edit")
    if edit:
        lines.append("")
        lines.append(f"**Proposed reward edit:** `{edit}`")
    return "\n".join(lines)


def load_frames() -> list[str]:
    frames = sorted(glob.glob(os.path.join(MOCK_DIR, "sample_frames", "*.png")))
    return frames


def load_reward_curve() -> str | None:
    path = os.path.join(MOCK_DIR, "reward_curve.png")
    return path if os.path.exists(path) else None


def refresh(run: int):
    frames = load_frames()
    curve = load_reward_curve()
    hyp = load_hypothesis_log(run)
    return frames, curve, hyp


with gr.Blocks(title="RL Policy Debugger") as demo:
    gr.Markdown("# RL Policy Debugger — Live Diagnosis")

    with gr.Row():
        run_selector = gr.Number(value=2, label="Run", precision=0)
        refresh_btn = gr.Button("Refresh")

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

    refresh_btn.click(
        fn=refresh,
        inputs=[run_selector],
        outputs=[rollout_gallery, reward_img, hypothesis_md],
    )

    demo.load(
        fn=refresh,
        inputs=[run_selector],
        outputs=[rollout_gallery, reward_img, hypothesis_md],
    )

if __name__ == "__main__":
    demo.launch()
