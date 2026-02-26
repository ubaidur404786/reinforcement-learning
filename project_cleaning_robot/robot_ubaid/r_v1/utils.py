"""
Visualization and helper utilities for the Cleaning Robot project (r_v1).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from .config import Config


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------

def render_grid(env, ax=None, figsize=(7, 7), title="Cleaning Robot"):
    """
    Render the current state of the environment as a color-coded grid.

    Parameters
    ----------
    env : RobotCleaningEnv
        Live environment instance.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (creates new figure if None).
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cfg = Config()
    rows, cols = cfg.grid_rows, cfg.grid_cols

    # Background
    grid_img = np.ones((rows, cols, 3))

    # Dirty tiles → brown
    for r in range(rows):
        for c in range(cols):
            if env.dirt_grid[r, c] == 1:
                grid_img[r, c] = [0.65, 0.45, 0.25]

    # Charging station → green
    cr, cc = cfg.charge_station
    grid_img[cr, cc] = [0.2, 0.8, 0.2]

    ax.imshow(grid_img, origin="upper")

    # Robot marker
    rx, ry = env.robot_pos
    ax.plot(ry, rx, "ro", markersize=14, label="Robot")

    # Grid lines
    for i in range(cols + 1):
        ax.axvline(i - 0.5, color="gray", linewidth=0.5)
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5)

    legend_elements = [
        mpatches.Patch(facecolor=[0.65, 0.45, 0.25], label="Dirty"),
        mpatches.Patch(facecolor=[0.2, 0.8, 0.2], label="Charging Station"),
        mpatches.Patch(facecolor="red", label="Robot"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_title(
        f"{title}\nBattery: {env.battery:.0f} | Step: {env.step_count} | "
        f"Dirty: {int(np.sum(env.dirt_grid))}",
        fontsize=10,
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Training progress plots
# ---------------------------------------------------------------------------

def plot_training_progress(metrics: dict, window: int = 20, figsize=(14, 10)):
    """
    Plot training metrics: rewards, episode lengths, dirt cleaned, epsilon.

    Parameters
    ----------
    metrics : dict
        Output of ``train.train()``.
    window : int
        Smoothing window for reward curve.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Episode rewards (smoothed)
    rewards = np.array(metrics["episode_rewards"])
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0, 0].plot(rewards, alpha=0.3, color="steelblue", label="Raw")
    axes[0, 0].plot(range(window - 1, len(rewards)), smoothed, color="steelblue",
                    linewidth=2, label=f"Smoothed (w={window})")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode lengths
    lengths = np.array(metrics["episode_lengths"])
    axes[0, 1].plot(lengths, color="darkorange", alpha=0.6)
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Dirt cleaned per episode
    dirt = np.array(metrics["dirt_cleaned"])
    axes[1, 0].plot(dirt, color="saddlebrown", alpha=0.6)
    axes[1, 0].set_title("Tiles Cleaned per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Tiles")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Epsilon decay
    eps = np.array(metrics["epsilon_values"])
    axes[1, 1].plot(eps, color="purple")
    axes[1, 1].set_title("Epsilon Decay")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Training Progress", fontsize=14)
    plt.tight_layout()
    return fig


def plot_q_heatmap(agent, battery_bin: int = None, action: int = None,
                   figsize=(10, 8)):
    """
    Plot a heatmap of Q-values averaged (or sliced) over battery bins.

    Parameters
    ----------
    agent : QLearningAgent
        Trained agent.
    battery_bin : int, optional
        Specific battery bin to display (None → average over all bins).
    action : int, optional
        Specific action (None → max over all actions).
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    if battery_bin is not None:
        q_slice = agent.q_table[:, :, battery_bin, :]
    else:
        q_slice = np.mean(agent.q_table, axis=2)

    if action is not None:
        heatmap = q_slice[:, :, action]
        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "CLEAN", "CHARGE"]
        title = f"Q-values – Action: {action_names[action]}"
    else:
        heatmap = np.max(q_slice, axis=2)
        title = "Max Q-value Heatmap"

    cmap = LinearSegmentedColormap.from_list(
        "qval", ["#2c3e50", "#3498db", "#2ecc71", "#f1c40f"]
    )
    im = ax.imshow(heatmap, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Q-value")

    cfg = Config()
    cr, cc = cfg.charge_station
    ax.plot(cc, cr, "w^", markersize=12, markeredgecolor="black", label="Charging Station")
    ax.legend(loc="upper right")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .environment import RobotCleaningEnv

    env = RobotCleaningEnv()
    env.reset(seed=0)

    fig, ax = render_grid(env, title="Initial State")
    plt.savefig("/tmp/grid_render_test.png", dpi=100)
    print("Grid render saved to /tmp/grid_render_test.png")
    plt.close()
