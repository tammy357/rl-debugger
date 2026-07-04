"""Run one evaluation episode from a saved checkpoint and export the exact
inputs Gemma (Stream 2) will receive: numbered frames + a reward-curve PNG,
plus a manifest tying them back to real sim/WandB step numbers.
"""

import argparse
import json
import os
import re

import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from env import PushEnv
from record_utils import plot_reward_curve, sample_frame_indices

STREAM1_DIR = os.path.dirname(__file__)
STEP_RANGE_WINDOW = 2500
CHECKPOINT_INTERVAL = 10_000


def _infer_checkpoint_step(checkpoint_path):
    """CheckpointCallback names files like model_10000_steps.zip; model_final
    has no step number, so caller must supply --checkpoint_step for that case."""
    match = re.search(r"_(\d+)_steps", os.path.basename(checkpoint_path))
    return int(match.group(1)) if match else None


def run_episode(model, env, deterministic=True, collect_frames=False, seed=None):
    """Runs one episode and returns its metrics (and optionally frames).

    Used both by run_rollout() (single flagship episode, deterministic action
    selection + a fixed seed so the clip is exactly reproducible run to run,
    frames on) and by run_trials.py's batch sweep (many stochastic episodes
    with no seed so each draws a fresh random start position, frames off --
    writing a PNG per step for hundreds of trial episodes would be wasteful).
    """
    obs, _ = env.reset(seed=seed)

    all_frames = []  # list of (sim_step, rgb array), only populated if collect_frames
    rewards = []
    drop_step = None
    success_step = None

    sim_step = 0
    while True:
        if collect_frames:
            all_frames.append((sim_step, env.get_camera_image()))
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))

        if info.get("dropped") and drop_step is None:
            drop_step = sim_step
        if info.get("success") and success_step is None:
            success_step = sim_step

        sim_step += 1
        if terminated or truncated:
            if collect_frames:
                all_frames.append((sim_step, env.get_camera_image()))
            break

    return {
        "frames": all_frames,
        "rewards": rewards,
        "num_episode_steps": sim_step,
        "drop_step": drop_step,
        "success_step": success_step,
        "success": success_step is not None,
        "total_reward": sum(rewards),
    }


def run_rollout(run_id, checkpoint_path, checkpoint_step=None, out_dir=None, episode_seed=0):
    if checkpoint_step is None:
        checkpoint_step = _infer_checkpoint_step(checkpoint_path)

    out_dir = out_dir or os.path.join(STREAM1_DIR, "outputs", f"run{run_id}")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    model = PPO.load(checkpoint_path)
    env = PushEnv()
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
        rewards, run_id=run_id, drop_step=drop_step, success_step=success_step, out_path=chart_path
    )

    step_range = None
    if checkpoint_step is not None:
        step_range = [max(0, checkpoint_step - STEP_RANGE_WINDOW), checkpoint_step + STEP_RANGE_WINDOW]

    manifest = {
        "run": run_id,
        "checkpoint_path": os.path.relpath(checkpoint_path, STREAM1_DIR),
        "checkpoint_step": checkpoint_step,
        "step_range": step_range,
        "num_episode_steps": n_total - 1,
        "drop_step": drop_step,
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
    parser.add_argument("--episode_seed", type=int, default=0,
                         help="seeds the episode's randomized start position so the flagship clip is reproducible")
    args = parser.parse_args()
    run_rollout(args.run_id, args.checkpoint, checkpoint_step=args.checkpoint_step, episode_seed=args.episode_seed)


if __name__ == "__main__":
    main()
