"""
Gymnasium-compatible Cleaning Robot Environment (r_v1).

Grid: 10x10
Charging station: (0, 0) – robot's start location.

Actions (Discrete 6):
    0 – UP    (row - 1)
    1 – DOWN  (row + 1)
    2 – LEFT  (col - 1)
    3 – RIGHT (col + 1)
    4 – CLEAN (clean current tile)
    5 – CHARGE (charge battery at station)

State: (x, y, battery_level)

Rewards:
    +10   cleaning a dirty tile
    -0.1  each movement step (battery cost)
    +50   returning to charging station after all tiles are cleaned
    -1    invalid action (move into wall, clean clean tile, charge away from station)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import Config


# Action indices
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
CLEAN = 4
CHARGE = 5

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "CLEAN", "CHARGE"]


class RobotCleaningEnv(gym.Env):
    """
    Cleaning robot on a 10×10 grid with battery management.

    The robot starts at the charging station (0, 0) fully charged.  It must
    visit every dirty tile, clean it, then return to the charging station.
    The episode ends when the robot returns to (0, 0) with all tiles clean,
    the battery hits 0, or MAX_STEPS is reached.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.cfg = Config()

        self.rows = self.cfg.grid_rows
        self.cols = self.cfg.grid_cols
        self.max_steps = self.cfg.max_steps
        self.render_mode = render_mode

        # Action and observation spaces
        self.action_space = spaces.Discrete(6)
        # Observation: [x, y, battery_level]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.rows - 1, self.cols - 1, self.cfg.max_battery], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (initialised in reset)
        self.robot_pos = None
        self.battery = None
        self.dirt_grid = None
        self.step_count = None
        self._all_clean_flag = False  # True once all tiles become clean

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = list(self.cfg.charge_station)
        self.battery = float(self.cfg.initial_battery)
        self.step_count = 0
        self._all_clean_flag = False

        # Randomly dirty ~30 % of tiles (excluding charging station)
        rng = np.random.default_rng(seed)
        self.dirt_grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != tuple(self.cfg.charge_station):
                    if rng.random() < 0.3:
                        self.dirt_grid[r, c] = 1

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        x, y = self.robot_pos

        if action in (UP, DOWN, LEFT, RIGHT):
            reward, terminated = self._handle_move(action)
        elif action == CLEAN:
            reward, terminated = self._handle_clean()
        elif action == CHARGE:
            reward, terminated = self._handle_charge()

        # Battery depletion check
        if self.battery <= 0:
            self.battery = 0.0
            terminated = True

        # Step limit
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        grid_str = self._render_ansi()
        print(grid_str)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_move(self, action):
        x, y = self.robot_pos
        dx, dy = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.rows and 0 <= ny < self.cols:
            self.robot_pos = [nx, ny]
            self.battery -= self.cfg.battery_move_cost
            reward = self.cfg.reward_move

            # Check: all clean and just returned to charging station?
            if self._is_at_station() and self._all_tiles_clean():
                reward += self.cfg.reward_all_cleaned_return
                return reward, True  # episode complete
        else:
            # Wall collision
            reward = self.cfg.reward_invalid

        return reward, False

    def _handle_clean(self):
        x, y = self.robot_pos
        if self.dirt_grid[x, y] == 1:
            self.dirt_grid[x, y] = 0
            self.battery -= self.cfg.battery_clean_cost
            reward = self.cfg.reward_clean_dirty

            # Mark flag if now all clean
            if self._all_tiles_clean():
                self._all_clean_flag = True
            return reward, False
        else:
            # Tile already clean
            return self.cfg.reward_invalid, False

    def _handle_charge(self):
        if self._is_at_station():
            self.battery = min(
                self.battery + self.cfg.battery_charge_rate,
                self.cfg.max_battery,
            )
            return 0.0, False
        else:
            return self.cfg.reward_invalid, False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_at_station(self):
        return tuple(self.robot_pos) == tuple(self.cfg.charge_station)

    def _all_tiles_clean(self):
        return int(np.sum(self.dirt_grid)) == 0

    def _get_obs(self):
        return np.array(
            [self.robot_pos[0], self.robot_pos[1], self.battery],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "step": self.step_count,
            "battery": self.battery,
            "dirt_remaining": int(np.sum(self.dirt_grid)),
            "at_station": self._is_at_station(),
            "all_clean": self._all_tiles_clean(),
        }

    def _render_ansi(self):
        lines = [
            f"Step: {self.step_count} | Battery: {self.battery:.0f} | "
            f"Dirt remaining: {int(np.sum(self.dirt_grid))}",
            "-" * (self.cols * 2 + 1),
        ]
        for r in range(self.rows):
            row = "|"
            for c in range(self.cols):
                if [r, c] == self.robot_pos:
                    row += "R|"
                elif (r, c) == tuple(self.cfg.charge_station):
                    row += "C|"
                elif self.dirt_grid[r, c] == 1:
                    row += "D|"
                else:
                    row += ".|"
            lines.append(row)
        lines.append("-" * (self.cols * 2 + 1))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = RobotCleaningEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print("Initial observation:", obs)
    print("Info:", info)

    total_reward = 0.0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nTotal reward after 20 random steps: {total_reward:.2f}")
    print("Final info:", info)
