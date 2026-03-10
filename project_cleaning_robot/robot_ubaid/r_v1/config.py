"""
Configuration file for the Cleaning Robot RL project (r_v1).
All hyperparameters and environment settings are defined here.
"""

# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------
GRID_ROWS = 10
GRID_COLS = 10

# Charging station (robot start position)
CHARGE_STATION = (0, 0)

# Battery limits
MAX_BATTERY = 100
INITIAL_BATTERY = 100
BATTERY_MOVE_COST = 1       # Battery consumed per movement step
BATTERY_CLEAN_COST = 5      # Battery consumed per clean action
BATTERY_CHARGE_RATE = 10    # Battery restored per charge step at station

# Episode limits
MAX_STEPS = 500

# ---------------------------------------------------------------------------
# Reward values
# ---------------------------------------------------------------------------
REWARD_CLEAN_DIRTY = 10.0        # Reward for cleaning a dirty tile
REWARD_MOVE = -0.1               # Penalty for each movement (battery cost)
REWARD_ALL_CLEANED_RETURN = 50.0 # Reward for returning to station with all tiles cleaned
REWARD_INVALID = -1.0            # Penalty for invalid actions

# ---------------------------------------------------------------------------
# Q-Learning hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 0.3
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995           # Multiplicative decay applied each episode

# Battery level discretisation for the state space
BATTERY_BINS = 10               # Number of discrete battery levels (0..9)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
N_EPISODES = 1000
CHECKPOINT_INTERVAL = 100       # Save Q-table every N episodes
CHECKPOINT_DIR = "checkpoints"

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
EVAL_EPISODES = 10

# ---------------------------------------------------------------------------
# Convenience class (allows attribute-style access)
# ---------------------------------------------------------------------------
class Config:
    """Container for all configuration constants."""
    grid_rows = GRID_ROWS
    grid_cols = GRID_COLS
    charge_station = CHARGE_STATION
    max_battery = MAX_BATTERY
    initial_battery = INITIAL_BATTERY
    battery_move_cost = BATTERY_MOVE_COST
    battery_clean_cost = BATTERY_CLEAN_COST
    battery_charge_rate = BATTERY_CHARGE_RATE
    max_steps = MAX_STEPS

    reward_clean_dirty = REWARD_CLEAN_DIRTY
    reward_move = REWARD_MOVE
    reward_all_cleaned_return = REWARD_ALL_CLEANED_RETURN
    reward_invalid = REWARD_INVALID

    learning_rate = LEARNING_RATE
    discount_factor = DISCOUNT_FACTOR
    epsilon_start = EPSILON_START
    epsilon_min = EPSILON_MIN
    epsilon_decay = EPSILON_DECAY
    battery_bins = BATTERY_BINS

    n_episodes = N_EPISODES
    checkpoint_interval = CHECKPOINT_INTERVAL
    checkpoint_dir = CHECKPOINT_DIR

    eval_episodes = EVAL_EPISODES
