"""
Q-Learning Agent with epsilon-greedy exploration (r_v1).

State space: (x, y, battery_bin) where battery_bin discretises the
continuous battery level into BATTERY_BINS equal-width buckets.

The design mirrors the proven approach from robot_alqa_v1 but extends the
state with a battery dimension and adds a CHARGE action (index 5).
"""

import numpy as np

from .config import Config


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration.

    Extended state: (x, y, battery_bin)

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    n_actions : int
        Number of discrete actions (default 6: UP/DOWN/LEFT/RIGHT/CLEAN/CHARGE).
    battery_bins : int
        Number of discrete battery levels for state discretisation.
    learning_rate : float
        Q-learning step size (alpha).
    discount_factor : float
        Future reward discount (gamma).
    epsilon : float
        Initial exploration rate.
    epsilon_decay : float
        Multiplicative decay applied to epsilon each episode.
    epsilon_min : float
        Minimum exploration rate.
    """

    def __init__(
        self,
        n_rows: int = 10,
        n_cols: int = 10,
        n_actions: int = 6,
        battery_bins: int = None,
        learning_rate: float = None,
        discount_factor: float = None,
        epsilon: float = None,
        epsilon_decay: float = None,
        epsilon_min: float = None,
    ):
        cfg = Config()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions
        self.battery_bins = battery_bins if battery_bins is not None else cfg.battery_bins
        self._max_battery = cfg.max_battery

        self.lr = learning_rate if learning_rate is not None else cfg.learning_rate
        self.gamma = discount_factor if discount_factor is not None else cfg.discount_factor
        self.epsilon = epsilon if epsilon is not None else cfg.epsilon_start
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else cfg.epsilon_decay
        self.epsilon_min = epsilon_min if epsilon_min is not None else cfg.epsilon_min

        # Q-table: shape (rows, cols, battery_bins, n_actions)
        self.q_table = np.zeros(
            (n_rows, n_cols, self.battery_bins, n_actions), dtype=np.float64
        )

        # Tracking
        self.exploration_history: list = []
        self.action_history: list = []
        self.episode_rewards: list = []

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------

    def _battery_bin(self, battery: float) -> int:
        """Map continuous battery level to a discrete bin index."""
        b = float(np.clip(battery, 0.0, self._max_battery))
        bin_idx = int(b / self._max_battery * self.battery_bins)
        return min(bin_idx, self.battery_bins - 1)

    def _discretise(self, obs) -> tuple:
        """Convert raw observation [x, y, battery] to (x, y, battery_bin)."""
        x, y, battery = int(obs[0]), int(obs[1]), float(obs[2])
        return x, y, self._battery_bin(battery)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, obs, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.

        Parameters
        ----------
        obs : array-like [x, y, battery]
            Current observation from the environment.
        training : bool
            When False, always exploit (greedy).

        Returns
        -------
        int
            Selected action index.
        """
        x, y, b = self._discretise(obs)

        if training and np.random.random() < self.epsilon:
            action = int(np.random.randint(self.n_actions))
            self.exploration_history.append(1)
        else:
            q_vals = self.q_table[x, y, b]
            max_q = np.max(q_vals)
            best = np.where(q_vals == max_q)[0]
            action = int(np.random.choice(best))
            self.exploration_history.append(0)

        self.action_history.append(action)
        return action

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, obs, action: int, reward: float, next_obs, done: bool = False):
        """
        Apply Q-learning update rule.

        Q(s,a) ← Q(s,a) + α * (r + γ * max_a' Q(s',a') − Q(s,a))
        """
        x, y, b = self._discretise(obs)
        nx, ny, nb = self._discretise(next_obs)

        current_q = self.q_table[x, y, b, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[nx, ny, nb])

        self.q_table[x, y, b, action] += self.lr * (target - current_q)

    # ------------------------------------------------------------------
    # Epsilon management
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Multiply epsilon by decay factor, respecting minimum."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_exploration_rate(self) -> float:
        """Return fraction of recent steps that were exploratory."""
        if not self.exploration_history:
            return 0.0
        return float(np.mean(self.exploration_history[-100:]))

    def reset_history(self):
        """Clear per-episode tracking lists."""
        self.exploration_history = []
        self.action_history = []

    def save_q_table(self, filepath: str):
        """Save Q-table (and action counts) to .npy files."""
        np.save(filepath, self.q_table)

    def load_q_table(self, filepath: str):
        """Load Q-table from a .npy file."""
        self.q_table = np.load(filepath)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .environment import RobotCleaningEnv

    env = RobotCleaningEnv()
    agent = QLearningAgent()

    obs, info = env.reset(seed=0)
    total_reward = 0.0

    for _ in range(50):
        action = agent.choose_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.learn(obs, action, reward, next_obs, terminated or truncated)
        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break

    agent.decay_epsilon()
    print(f"Smoke-test reward: {total_reward:.2f}")
    print(f"Epsilon after decay: {agent.epsilon:.4f}")
    print(f"Exploration rate: {agent.get_exploration_rate():.2f}")
