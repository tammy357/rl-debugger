"""Standalone rollout exporter for the bonus cart-pole scenario. Mirrors
rollout.py's run_rollout() orchestration but points at CartPoleBalanceEnv
instead of PushEnv, since run_rollout() hardcodes PushEnv and touching
rollout.py is out of scope. Reuses run_episode (duck-typed -- only touches
env.reset/step/get_camera_image + model.predict) and
sample_frame_indices/plot_reward_curve unchanged.
"""

import argparse
import json
import os

from PIL import Image
from stable_baselines3 import PPO

from cartpole_env import CartPoleBalanceEnv
from rollout import run_episode, _infer_checkpoint_step
from record_utils import plot_reward_curve, sample_frame_indices

STREAM1_DIR = os.path.dirname(__file__)
STEP_RANGE_WINDOW = 2500


def run_cartpole_rollout(run_id, checkpoint_path, checkpoint_step=None, out_dir=None, episode_seed=0):
    if checkpoint_step is None:
        checkpoint_step = _infer_checkpoint_step(checkpoint_path)

    out_dir = out_dir or os.path.join(STREAM1_DIR, "outputs", f"run{run_id}")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    model = PPO.load(checkpoint_path)
    env = CartPoleBalanceEnv()
    episode = run_episode(model, env, deterministic=True, collect_frames=True, seed=episode_seed)
    env.close()

    all_frames = episode["frames"]
    rewards = episode["rewards"]
    drop_step = episode["drop_step"]
    success_step = episode["success_step"]

    n_total = len(all_frames)
    keep_indices = sample_frame_indices(n_total, drop_step=drop_step)

    frame_records = []
    for out_idx, frame_i in enumerate(keep_indices):
        sim_i, rgb = all_frames[frame_i]
        fname = f"frame_{out_idx:04d}.png"
        Image.fromarray(rgb).save(os.path.join(frames_dir, fname))
        frame_records.append({"file": f"frames/{fname}", "sim_step": sim_i})

    chart_path = os.path.join(out_dir, "reward_curve.png")
    plot_reward_curve(
        rewards, run_id=run_id, drop_step=drop_step, success_step=success_step,
        out_path=chart_path, event_label="pole fell",
    )

    step_range = None
    if checkpoint_step is not None:
        step_range = [max(0, checkpoint_step - STEP_RANGE_WINDOW), checkpoint_step + STEP_RANGE_WINDOW]

    manifest = {
        "run": run_id,
        "task": "cartpole_balance",
        "checkpoint_path": os.path.relpath(checkpoint_path, STREAM1_DIR),
        "checkpoint_step": checkpoint_step,
        "step_range": step_range,
        "num_episode_steps": n_total - 1,
        "drop_step": drop_step,
        "pole_fall_step": drop_step,
        "success": episode["success"],
        "total_reward": episode["total_reward"],
        "frames": frame_records,
        "reward_curve": "reward_curve.png",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"run{run_id}: {len(frame_records)} frames, drop_step={drop_step}, success={manifest['success']}, "
          f"total_reward={manifest['total_reward']:.1f}, step_range={step_range}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--episode_seed", type=int, default=0)
    args = parser.parse_args()
    run_cartpole_rollout(
        args.run_id, args.checkpoint, checkpoint_step=args.checkpoint_step, episode_seed=args.episode_seed
    )


if __name__ == "__main__":
    main()
