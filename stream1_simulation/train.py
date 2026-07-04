"""Train PPO on PushEnv and save periodic checkpoints.

Each --run_id is an independent training session (own seed, own WandB run,
own global step counter from 0) -- this is what makes Stream 3's per-run
step_range references meaningful.
"""

import argparse
import os

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from env import PushEnv

STREAM1_DIR = os.path.dirname(__file__)
CHECKPOINT_INTERVAL = 10_000


def train_one(run_id, seed, timesteps=100_000, wandb_mode="offline"):
    """Trains one PPO policy for `run_id`/`seed` and returns the checkpoint dir.

    Extracted from main() so run_trials.py's batch sweep can call this
    in-process for many seeds without spawning a subprocess per seed.
    """
    checkpoint_dir = os.path.join(STREAM1_DIR, "checkpoints", f"run{run_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    run = wandb.init(
        project="rl-policy-debugger",
        name=f"run{run_id}",
        group="stream1-training",
        mode=wandb_mode,
        sync_tensorboard=True,
        config={"run_id": run_id, "seed": seed, "timesteps": timesteps},
    )

    env = PushEnv()
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        tensorboard_log=os.path.join(STREAM1_DIR, "tb_logs", f"run{run_id}"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_INTERVAL,
        save_path=checkpoint_dir,
        name_prefix="model",
    )

    from wandb.integration.sb3 import WandbCallback

    callbacks = [
        checkpoint_callback,
        WandbCallback(verbose=1),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(os.path.join(checkpoint_dir, "model_final"))
    env.close()
    run.finish()
    return checkpoint_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument(
        "--wandb_mode",
        default=os.environ.get("WANDB_MODE", "offline"),
        choices=["online", "offline", "disabled"],
        help="defaults to offline so training never blocks on network/login",
    )
    args = parser.parse_args()
    train_one(args.run_id, args.seed, timesteps=args.timesteps, wandb_mode=args.wandb_mode)


if __name__ == "__main__":
    main()
