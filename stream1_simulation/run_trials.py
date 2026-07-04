"""Batch sweep: train many seeds and roll out many stochastic episodes per
checkpoint, to measure a real bug-reproduction rate instead of relying on a
handful of hand-picked demo episodes.

Each episode uses a randomized start position (see env.py's reset()) and
stochastic action sampling (run_episode(..., deterministic=False)), so
repeated rollouts of the same checkpoint are genuinely different trials.

Usage:
    python run_trials.py --num_seeds 15 --timesteps 100000 --episodes_per_checkpoint 20

Appends one JSON line per episode to outputs/trial_log.jsonl (append mode, so
a sweep that's interrupted partway through doesn't lose earlier seeds' data).
Run analyze_trials.py afterward to aggregate this into bug-rate statistics.
"""

import argparse
import json
import os

from stable_baselines3 import PPO

from env import PushEnv
from rollout import run_episode
from train import train_one, CHECKPOINT_INTERVAL

STREAM1_DIR = os.path.dirname(__file__)
TRIAL_LOG_PATH = os.path.join(STREAM1_DIR, "outputs", "trial_log.jsonl")


def checkpoint_paths(run_id, timesteps):
    """Yields (checkpoint_step, path) for every checkpoint saved during
    training, including the final policy (checkpoint_step=timesteps).

    model_final.zip is only yielded separately when timesteps isn't a
    multiple of CHECKPOINT_INTERVAL -- otherwise it's the exact same trained
    policy as the last periodic checkpoint, and evaluating both would just
    double-count that step's episodes for no new information.
    """
    checkpoint_dir = os.path.join(STREAM1_DIR, "checkpoints", f"run{run_id}")
    for step in range(CHECKPOINT_INTERVAL, timesteps + 1, CHECKPOINT_INTERVAL):
        yield step, os.path.join(checkpoint_dir, f"model_{step}_steps.zip")
    if timesteps % CHECKPOINT_INTERVAL != 0:
        yield timesteps, os.path.join(checkpoint_dir, "model_final.zip")


def run_sweep(num_seeds, timesteps, episodes_per_checkpoint, seed_start, wandb_mode):
    os.makedirs(os.path.dirname(TRIAL_LOG_PATH), exist_ok=True)

    with open(TRIAL_LOG_PATH, "a") as log_file:
        for run_id in range(seed_start, seed_start + num_seeds):
            print(f"=== seed {run_id}: training for {timesteps} steps ===")
            train_one(run_id, seed=run_id, timesteps=timesteps, wandb_mode=wandb_mode)

            env = PushEnv()
            for checkpoint_step, checkpoint_path in checkpoint_paths(run_id, timesteps):
                if not os.path.exists(checkpoint_path):
                    print(f"  [seed {run_id}] missing checkpoint at step {checkpoint_step}, skipping")
                    continue
                model = PPO.load(checkpoint_path)
                n_dropped = 0
                for _ in range(episodes_per_checkpoint):
                    episode = run_episode(model, env, deterministic=False, collect_frames=False)
                    record = {
                        "seed": run_id,
                        "checkpoint_step": checkpoint_step,
                        "dropped": episode["drop_step"] is not None,
                        "success": episode["success"],
                        "drop_step": episode["drop_step"],
                        "total_reward": episode["total_reward"],
                    }
                    n_dropped += record["dropped"]
                    log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(f"  [seed {run_id}] step {checkpoint_step}: "
                      f"{n_dropped}/{episodes_per_checkpoint} dropped")
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes_per_checkpoint", type=int, default=20)
    parser.add_argument(
        "--seed_start", type=int, default=4,
        help="first run_id/seed to use -- default 4 so it doesn't clobber the existing run1-3 checkpoints",
    )
    parser.add_argument(
        "--wandb_mode", default="disabled", choices=["online", "offline", "disabled"],
        help="defaults to disabled: a 15+ seed sweep would otherwise create a WandB offline-run dir per seed",
    )
    args = parser.parse_args()
    run_sweep(args.num_seeds, args.timesteps, args.episodes_per_checkpoint, args.seed_start, args.wandb_mode)


if __name__ == "__main__":
    main()
