"""
Training loop for the Cleaning Robot RL project (r_v1).

Usage
-----
    python -m project_cleaning_robot.robot_ubaid.r_v1.train
    # or from the r_v1 directory:
    python train.py
"""

import os
import numpy as np

from .environment import RobotCleaningEnv
from .agent import QLearningAgent
from .config import Config


def train(n_episodes: int = None, seed: int = 42) -> tuple:
    """
    Run the full training loop.

    Parameters
    ----------
    n_episodes : int, optional
        Override the number of training episodes from Config.
    seed : int
        Base random seed.

    Returns
    -------
    agent : QLearningAgent
        Trained agent.
    metrics : dict
        Dictionary with lists: episode_rewards, episode_lengths,
        dirt_cleaned, epsilon_values.
    """
    cfg = Config()
    n_episodes = n_episodes or cfg.n_episodes

    env = RobotCleaningEnv()
    agent = QLearningAgent()

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "dirt_cleaned": [],
        "epsilon_values": [],
    }

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"Training for {n_episodes} episodes...")
    print(f"Grid: {cfg.grid_rows}x{cfg.grid_cols} | "
          f"Max battery: {cfg.max_battery} | "
          f"Max steps/ep: {cfg.max_steps}")
    print("-" * 60)

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        steps = 0

        while True:
            action = agent.choose_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.learn(obs, action, reward, next_obs, done)

            episode_reward += reward
            obs = next_obs
            steps += 1

            if done:
                break

        # Decay exploration after each episode
        agent.decay_epsilon()

        # Record metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(steps)
        metrics["dirt_cleaned"].append(
            cfg.grid_rows * cfg.grid_cols - info["dirt_remaining"]
        )
        metrics["epsilon_values"].append(agent.epsilon)

        # Progress logging
        if episode % 100 == 0:
            recent = metrics["episode_rewards"][-100:]
            print(
                f"Episode {episode:5d}/{n_episodes} | "
                f"Avg reward (last 100): {np.mean(recent):8.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Steps: {steps:4d}"
            )

        # Checkpoint
        if episode % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"q_table_ep{episode}.npy")
            agent.save_q_table(ckpt_path)

    # Save final Q-table
    final_path = os.path.join(cfg.checkpoint_dir, "q_table_final.npy")
    agent.save_q_table(final_path)
    print(f"\nTraining complete. Q-table saved to '{final_path}'.")

    return agent, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent, metrics = train()

    print("\n=== Training Summary ===")
    rewards = metrics["episode_rewards"]
    print(f"Total episodes : {len(rewards)}")
    print(f"Mean reward    : {np.mean(rewards):.2f}")
    print(f"Best episode   : {np.max(rewards):.2f}")
    print(f"Final epsilon  : {metrics['epsilon_values'][-1]:.4f}")
