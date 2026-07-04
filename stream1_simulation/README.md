# Stream 1 — Simulation & Data Generation

PyBullet pushing task with a deliberately buggy reward (missing penalty for
dropping the object off the table), three independently-trained PPO policies,
and a rollout recorder that exports the exact inputs Stream 2's
`analyze_run(frames, chart, hypothesis_log)` expects.

## The task

A sphere "pusher" (velocity-controlled, xy-only) pushes a cube toward a target
zone placed near the table's front edge. See `env.py` for exact geometry/constants.

## The bug

In `env.py`'s `step()`: the object falling off the table (`obj_z` more than
`DROP_MARGIN` below the table top) is detected and used to end the episode,
but is never subtracted from `reward`. A distance-minimizing policy has no
disincentive against overshooting the target and shoving the object off.

## Setup

```bash
cd stream1_simulation
python3.11 -m venv ../venv   # see note below on Python version
source ../venv/bin/activate
pip install pybullet stable-baselines3 gymnasium matplotlib wandb tensorboard Pillow
```

**Apple Silicon macOS note:** `pip install pybullet` may fail to build from
source with a clang error in the bundled zlib (`fdopen` macro collides with
the macOS SDK's `_stdio.h` on newer Xcode Command Line Tools). No prebuilt
wheel exists for macOS arm64 on PyPI. Fix: download the sdist
(`pip download --no-binary :all: --no-deps pybullet`), patch
`examples/ThirdPartyLibs/zlib/zutil.h` line ~121 from
`#if defined(MACOS) || defined(TARGET_OS_MAC)` to
`#if (defined(MACOS) || defined(TARGET_OS_MAC)) && !defined(__APPLE__)`,
then `pip wheel --no-deps -w wheel_out .` and install the resulting wheel.
Also use Homebrew's Python 3.11 rather than system Python 3.9 if available.

WandB defaults to **offline mode** (no login/network required); set
`WANDB_MODE=online` once a real account is wired up.

## Usage

```bash
python sanity_check.py                                     # <30s smoke test, no training
python train.py --run_id 1 --seed 1 --timesteps 100000      # trains + checkpoints every 10k steps
python rollout.py --run_id 1 --checkpoint checkpoints/run1/model_80000_steps.zip --checkpoint_step 80000
python check_gemma_contract.py --run_id 1                   # validates output shape
```

`train.py` saves checkpoints every 10k steps. PPO on this task tends to learn
to push more aggressively over training (overshooting and dropping the object
— the bug — increasingly often from ~step 60k on). Measured across 15 seeds x
20 stochastic rollouts/checkpoint (see Trial statistics below), the bug
reproduction rate rises from 0% at step 50k to 54% by step 100k and plateaus
there — it does **not** self-correct back down with more training. (An
earlier version of this doc claimed 2 of 3 seeds "self-correct into a clean
success behavior" by ~80k-90k; that was based on eyeballing one deterministic
episode per checkpoint per seed, which the real aggregate data does not
support once episodes actually vary from run to run.)

**Pick the demo checkpoint using measured data, not eyeballing.** Run
`run_trials.py` + `analyze_trials.py` (below) to get a real bug-reproduction
rate per checkpoint step, aggregated across many seeds and many stochastic
rollouts each — then pick a checkpoint step with a high measured rate for the
flagship demo episode, backed by a number like "83% across 300 episodes"
instead of "this one episode looked right."

## Trial statistics (`run_trials.py` + `analyze_trials.py`)

The 3 flagship runs below are one hand-picked episode each — good for the
demo's qualitative rollout-video panel, but not itself evidence that the bug
generalizes. `run_trials.py` trains many additional seeds and, for every
checkpoint, runs many stochastic episodes with randomized start positions
(see `env.py`'s `reset()`) so repeated rollouts of the same checkpoint are
genuinely different trials rather than a replay of the same episode:

```bash
python run_trials.py --num_seeds 15 --timesteps 100000 --episodes_per_checkpoint 20
python analyze_trials.py
```

This appends every episode's outcome to `outputs/trial_log.jsonl` and produces
`outputs/trial_summary.json` (per-checkpoint bug rate / success rate / mean
reward, aggregated across all seeds) and `outputs/bug_rate_curve.png` (bug
reproduction rate vs. checkpoint step) — turning "2 of 3 seeds self-correct"
from an anecdote into a measured curve over dozens of seeds and hundreds of
episodes per checkpoint. Each run of `run_trials.py` trains fresh seeds
starting at `--seed_start` (default 4), so it never touches the existing
run1-3 checkpoints/outputs below.

## Output layout

```
outputs/run{N}/
├── frames/frame_0000.png ...   # 320x320 RGB, evenly-sampled + drop-step forced in
├── reward_curve.png            # matplotlib, dataviz-skill styled
└── manifest.json                # checkpoint_step, step_range, drop_step, frame index -> sim step
```

`manifest.json`'s `step_range` is a ±2500-step window around the checkpoint's
real training step — this is the number Stream 3/4 should use so Computer
Use's WandB screenshot actually lines up with logged data.

## Current status (demo assets)

| Run | Checkpoint | Behavior | Drop step |
|---|---|---|---|
| 1 | `model_80000_steps` | drops the object | 65 |
| 2 | `model_100000_steps` | drops the object | 63 |
| 3 | `model_70000_steps` | drops the object | 64 |

All three sets pass `check_gemma_contract.py`. **Note:** these 3 checkpoints
were the original hand-eyeballed selection (pre-dating the randomized-start
env and the trial-statistics harness above) — they still work as Gemma's
input contract, but weren't chosen using the measured bug-rate data. Left
in place as-is since other streams may already reference these exact paths.

## Evidence-selected runs (4-6)

Runs 4-6 are the same `manifest.json`/`frames`/`reward_curve.png` shape as 1-3,
but each checkpoint was chosen from `run_trials.py`'s sweep data (already-
trained seeds 4-18) based on a high *measured* stochastic bug rate at that
step, then confirmed to reproduce the bug under a fixed, reproducible
`--episode_seed` (deterministic action selection alone isn't enough to
reproduce a clip run-to-run anymore now that start position is randomized --
`rollout.py`'s `--episode_seed` pins that randomness). They also span the
bug's emergence curve rather than all looking like near-duplicates:

| Run | Checkpoint (seed) | Checkpoint step | Measured stochastic bug rate at this step | Drop step |
|---|---|---|---|---|
| 4 | seed 7, `model_60000_steps` | 60,000 | 20% (early emergence) | 66 |
| 5 | seed 11, `model_80000_steps` | 80,000 | 70% (growing) | 63 |
| 6 | seed 5, `model_100000_steps` | 100,000 | 90% (strong evidence) | 65 |

Regenerate with (checkpoints already exist from the sweep, so this only
re-runs the deterministic rollout, not training):

```bash
python rollout.py --run_id 4 --checkpoint checkpoints/run7/model_60000_steps.zip --checkpoint_step 60000 --episode_seed 0
python rollout.py --run_id 5 --checkpoint checkpoints/run11/model_80000_steps.zip --checkpoint_step 80000 --episode_seed 0
python rollout.py --run_id 6 --checkpoint checkpoints/run5/model_100000_steps.zip --checkpoint_step 100000 --episode_seed 3
```

## Files

- `env.py`: defines `PushEnv`, the PyBullet pushing 
task and its deliberately buggy reward function, 
imported by every other script below.
- `train.py`: trains a PPO policy on `PushEnv` and 
saves periodic checkpoints, logging to WandB.
- `rollout.py`: loads a checkpoint saved by `train.py`
and runs it through `PushEnv` to produce a run's frames, 
reward-curve chart, and `manifest.json`.
- `record_utils.py`: frame-sampling and reward-curve-plotting
helpers used by `rollout.py`.
- `sanity_check.py`: runs a random policy through `PushEnv` 
as a <30s smoke test of the physics and camera, meant to 
catch env bugs before spending time on a full `train.py` run.
- `check_gemma_contract.py`: validates that a run's `rollout.py`
output matches the shape Stream 2's 
`analyze_run(frames, chart, hypothesis_log)` expects.
