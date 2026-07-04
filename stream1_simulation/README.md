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

`train.py` saves checkpoints every 10k steps. **Pick the demo checkpoint
manually**: PPO on this task tends to first learn to push aggressively
(overshooting and dropping the object — the bug, ~steps 60k-80k) before later
self-correcting into a clean, careful "success" behavior. Scan a run's
checkpoints and eyeball which one best demonstrates the bug; that's the one to
roll out for the demo. (Whether/when self-correction happens varies by seed —
run2's policy never fully self-corrected within 100k steps and reproduces the
bug at every checkpoint from 60k on; run1 and run3 self-correct by ~80k-90k.)

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

All three sets pass `check_gemma_contract.py`.

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
