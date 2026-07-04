"""Frame sampling and reward-curve plotting shared by rollout.py."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SAMPLED_FRAMES = 14

# dataviz skill palette (see stream1_simulation/README.md for the convention).
COLOR_SURFACE = "#fcfcfb"
COLOR_SERIES = "#2a78d6"
COLOR_CRITICAL = "#d03b3b"
COLOR_GOOD = "#0ca30c"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"


def sample_frame_indices(n_total_steps, drop_step=None, n_frames=N_SAMPLED_FRAMES):
    """Evenly-spaced frame indices across the episode, with the drop step
    force-included (deduplicated) so the fall is never skipped."""
    if n_total_steps <= n_frames:
        indices = list(range(n_total_steps))
    else:
        indices = list(np.linspace(0, n_total_steps - 1, n_frames, dtype=int))
    if drop_step is not None and drop_step not in indices:
        indices.append(drop_step)
    return sorted(set(indices))


def plot_reward_curve(rewards, run_id, drop_step=None, success_step=None, out_path=None,
                       event_label="object dropped"):
    """Save a reward-curve PNG for one rollout, styled per the dataviz skill:
    single series (no legend, title names it), status-colored event annotations,
    text always in ink/muted tokens rather than the series color."""
    steps = np.arange(len(rewards))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    ax.plot(steps, rewards, color=COLOR_SERIES, linewidth=2)

    ax.grid(True, color=COLOR_GRID, linewidth=1, axis="y")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)

    ax.ticklabel_format(useOffset=False, style="plain", axis="both")
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax.set_xlabel("rollout step", color=COLOR_MUTED, fontsize=10)
    ax.set_ylabel("reward", color=COLOR_MUTED, fontsize=10)
    ax.set_title(f"Run {run_id} — Rollout Reward Curve", color=COLOR_INK, fontsize=13, loc="left")

    ymin, ymax = ax.get_ylim()

    if drop_step is not None:
        ax.axvline(drop_step, color=COLOR_CRITICAL, linestyle="--", linewidth=1.5)
        ax.text(
            drop_step,
            ymax - 0.05 * (ymax - ymin),
            f"  {event_label}, step {drop_step}",
            color=COLOR_INK,
            fontsize=9,
            va="top",
        )

    if success_step is not None:
        ax.axvline(success_step, color=COLOR_GOOD, linestyle="--", linewidth=1.5)
        ax.text(
            success_step,
            ymin + 0.05 * (ymax - ymin),
            f"  target reached, step {success_step}",
            color=COLOR_INK,
            fontsize=9,
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_SURFACE)
    plt.close(fig)
    return out_path


def plot_bug_rate_curve(checkpoint_steps, bug_rates, n_seeds, n_episodes_per_checkpoint, out_path=None):
    """Save a bug-reproduction-rate-vs-checkpoint-step PNG, aggregated across
    many seeds/episodes (see analyze_trials.py), styled to match
    plot_reward_curve's dataviz-skill convention."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    ax.plot(checkpoint_steps, bug_rates, color=COLOR_CRITICAL, linewidth=2, marker="o", markersize=4)

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, color=COLOR_GRID, linewidth=1, axis="y")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)

    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax.set_xlabel("checkpoint step", color=COLOR_MUTED, fontsize=10)
    ax.set_ylabel("bug reproduction rate", color=COLOR_MUTED, fontsize=10)
    ax.set_title(
        f"Bug reproduction rate — {n_seeds} seeds, "
        f"{n_episodes_per_checkpoint} rollouts/checkpoint",
        color=COLOR_INK, fontsize=12, loc="left",
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_SURFACE)
    plt.close(fig)
    return out_path
