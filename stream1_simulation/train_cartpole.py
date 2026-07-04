"""Standalone PPO trainer for the bonus cart-pole scenario (run 7-10 in the
demo). Deliberately does not touch train.py -- writes to its own
checkpoints/bonus_cartpole/seed{N}/ and tb_logs/bonus_cartpole/seed{N}/
directories so it can never collide with the run1-18 checkpoints/tb_logs
already used by the flagship pusher runs and run_trials.py's seed sweep.
"""

import argparse
import os

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

from cartpole_env import CartPoleBalanceEnv

STREAM1_DIR = os.path.dirname(__file__)
CHECKPOINT_INTERVAL = 10_000


def train_cartpole(seed, timesteps=100_000, wandb_mode="offline"):
    checkpoint_dir = os.path.join(STREAM1_DIR, "checkpoints", "bonus_cartpole", f"seed{seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    run = wandb.init(
        project="rl-policy-debugger",
        name=f"bonus_cartpole_seed{seed}",
        group="stream1-bonus-cartpole",
        mode=wandb_mode,
        sync_tensorboard=True,
        config={"seed": seed, "timesteps": timesteps, "task": "cartpole_balance"},
    )

    env = CartPoleBalanceEnv()
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        tensorboard_log=os.path.join(STREAM1_DIR, "tb_logs", "bonus_cartpole", f"seed{seed}"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_INTERVAL,
        save_path=checkpoint_dir,
        name_prefix="model",
    )
    callbacks = [checkpoint_callback, WandbCallback(verbose=1)]

    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(os.path.join(checkpoint_dir, "model_final"))
    env.close()
    run.finish()
    return checkpoint_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument(
        "--wandb_mode",
        default=os.environ.get("WANDB_MODE", "offline"),
        choices=["online", "offline", "disabled"],
    )
    args = parser.parse_args()
    train_cartpole(args.seed, timesteps=args.timesteps, wandb_mode=args.wandb_mode)


if __name__ == "__main__":
    main()
