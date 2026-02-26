"""
Evaluation script for the trained Cleaning Robot agent (r_v1).

Usage
-----
    python -m project_cleaning_robot.robot_ubaid.r_v1.evaluate
    # or from the r_v1 directory:
    python evaluate.py
"""

import os
import numpy as np

from .environment import RobotCleaningEnv
from .agent import QLearningAgent
from .config import Config


def evaluate(
    q_table_path: str = None,
    n_episodes: int = None,
    render: bool = False,
    seed: int = 0,
) -> dict:
    """
    Evaluate a trained agent on the cleaning environment.

    Parameters
    ----------
    q_table_path : str, optional
        Path to saved Q-table (.npy).  Defaults to
        ``<checkpoint_dir>/q_table_final.npy``.
    n_episodes : int, optional
        Number of evaluation episodes (defaults to Config.eval_episodes).
    render : bool
        Print the grid each step when True.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    dict
        Evaluation metrics: episode_rewards, episode_lengths,
        dirt_cleaned, success_rate.
    """
    cfg = Config()
    n_episodes = n_episodes or cfg.eval_episodes

    if q_table_path is None:
        q_table_path = os.path.join(cfg.checkpoint_dir, "q_table_final.npy")

    agent = QLearningAgent()
    if os.path.isfile(q_table_path):
        agent.load_q_table(q_table_path)
        print(f"Loaded Q-table from '{q_table_path}'")
    else:
        print(f"No Q-table found at '{q_table_path}'. Using untrained agent.")

    env = RobotCleaningEnv(render_mode="human" if render else None)

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "dirt_cleaned": [],
        "success_rate": 0.0,
    }

    successes = 0

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        steps = 0

        while True:
            action = agent.choose_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # Count success: all tiles cleaned and returned to station
        if info["all_clean"] and info["at_station"]:
            successes += 1

        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(steps)
        metrics["dirt_cleaned"].append(
            cfg.grid_rows * cfg.grid_cols - info["dirt_remaining"]
        )

        print(
            f"Episode {episode:3d} | Reward: {episode_reward:8.2f} | "
            f"Steps: {steps:4d} | "
            f"Dirt cleaned: {metrics['dirt_cleaned'][-1]} | "
            f"All clean: {info['all_clean']}"
        )

    metrics["success_rate"] = successes / n_episodes

    print("\n=== Evaluation Summary ===")
    print(f"Episodes     : {n_episodes}")
    print(f"Mean reward  : {np.mean(metrics['episode_rewards']):.2f}")
    print(f"Mean steps   : {np.mean(metrics['episode_lengths']):.1f}")
    print(f"Success rate : {metrics['success_rate'] * 100:.1f}%")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(render=False)
