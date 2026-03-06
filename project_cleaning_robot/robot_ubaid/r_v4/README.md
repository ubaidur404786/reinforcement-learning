# 🤖 Cleaning Robot — Reinforcement Learning

A beginner-friendly project that trains a virtual cleaning robot to clean a house using **three different Reinforcement Learning algorithms**: **Q-Learning**, **SARSA**, and **DQN (Deep Q-Network)**. Watch the robot go from completely clueless to a cleaning expert — all through trial and error!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-orange)

---

## 📖 Table of Contents

- [What Is This Project?](#-what-is-this-project)
- [What Will You Learn?](#-what-will-you-learn)
- [The Environment — A House to Clean](#-the-environment--a-house-to-clean)
  - [House Layout](#house-layout)
  - [What the Robot Can Do (Actions)](#what-the-robot-can-do-actions)
  - [How the Robot "Sees" the World (State)](#how-the-robot-sees-the-world-state)
  - [Rewards & Penalties](#rewards--penalties)
- [The Algorithms — Three Ways to Learn](#-the-algorithms--three-ways-to-learn)
  - [1. Q-Learning (Off-Policy, Tabular)](#1-q-learning-off-policy-tabular)
  - [2. SARSA (On-Policy, Tabular)](#2-sarsa-on-policy-tabular)
  - [3. DQN — Deep Q-Network (Deep RL)](#3-dqn--deep-q-network-deep-rl)
  - [When Should I Use Which Algorithm?](#-when-should-i-use-which-algorithm)
  - [Quick Comparison Table](#quick-comparison-table)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Features](#-features)
- [How Training Works](#-how-training-works)
- [Results & Comparison](#-results--comparison)
- [Key Concepts for Beginners](#-key-concepts-for-beginners)
- [License](#-license)

---

## 🧹 What Is This Project?

Imagine you have a robot vacuum cleaner, but instead of programming it step-by-step ("go left, then clean, then go right..."), you let it **figure out how to clean on its own** through trial and error. That's exactly what this project does!

The robot starts with **zero knowledge** about:

- Where the rooms are
- Which tiles are dirty
- What actions lead to rewards

Through thousands of training episodes, the robot learns an optimal cleaning strategy entirely on its own. This is the magic of **Reinforcement Learning (RL)**.

### The Learning Loop

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   Robot observes state ──► Picks an action       │
│         ▲                       │                │
│         │                       ▼                │
│   Updates its brain ◄── Gets reward/penalty      │
│   (Q-table or Neural Net)                        │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 🎓 What Will You Learn?

By exploring this project, you'll understand:

| Concept                         | Description                                     |
| ------------------------------- | ----------------------------------------------- |
| **Reinforcement Learning**      | How agents learn from rewards, not instructions |
| **Q-Learning**                  | The classic off-policy tabular RL algorithm     |
| **SARSA**                       | The on-policy alternative to Q-Learning         |
| **DQN**                         | Using neural networks to approximate Q-values   |
| **Exploration vs Exploitation** | The fundamental RL trade-off                    |
| **Custom Gym Environments**     | Building your own environments with Gymnasium   |
| **Pygame Visualization**        | Watching your trained agent in action           |

---

## 🏠 The Environment — A House to Clean

The robot lives in a house made up of an **8×6 grid** with three rooms. Its mission: clean every dirty tile as efficiently as possible.

### House Layout

```
    Col:  0     1     2     3     4     5     6     7
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Row 0:   │WALL │WALL │WALL │WALL │WALL │WALL │WALL │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 1:   │WALL │ 🟡  │ 🟡  │ 🟡  │WALL │ 🔵  │ 🔵  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 2:   │WALL │ 🟡  │ 🟡  │ 🟡  │ ⬜  │ 🔵  │ 🔵  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 3:   │WALL │ 🟡  │ 🟡  │ 🟡  │ ⬜  │ 🔵  │ 🔵  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 4:   │WALL │ ⬜  │ ⬜  │ ⬜  │ ⬜  │ ⬜  │ ⬜  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 5:   │WALL │WALL │WALL │WALL │WALL │WALL │WALL │WALL │
         └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

🟡 Kitchen (9 tiles)  — Highest priority, +50 reward
🔵 Living Room (6 tiles) — Medium priority, +35 reward
⬜ Hallway (8 tiles)  — Lowest priority, +20 reward
```

**Total cleanable tiles: 23**

### What the Robot Can Do (Actions)

The robot has **6 possible actions** at each step:

| Action | Name         | What It Does                  |
| ------ | ------------ | ----------------------------- |
| 0      | **Forward**  | Move up (row decreases)       |
| 1      | **Backward** | Move down (row increases)     |
| 2      | **Left**     | Move left (column decreases)  |
| 3      | **Right**    | Move right (column increases) |
| 4      | **Wait**     | Stay in place (penalized!)    |
| 5      | **Clean**    | Clean the current tile        |

### How the Robot "Sees" the World (State)

The state encodes everything the robot needs to make decisions:

```
State = Position × Dirt Status × Movement History × DNUT Direction
         (23)       (2)            (5)                (10)

Total: 23 × 2 × 5 × 10 = 2,300 possible states
```

| Component            | Values                      | Purpose                                  |
| -------------------- | --------------------------- | ---------------------------------------- |
| **Position**         | 23 cleanable tile positions | Where is the robot?                      |
| **Dirt Status**      | Clean (0) or Dirty (1)      | Is the current tile dirty?               |
| **Movement History** | N / S / E / W / None        | Which direction did the robot come from? |
| **DNUT Direction**   | 3×3 grid directions + None  | Compass hint toward nearest dirty tile   |

> **DNUT** = Detection of Nearest Uncleaned Tile — gives the robot a compass-like hint pointing toward the closest dirty tile.

### Rewards & Penalties

The reward system is what **teaches** the robot. Good actions get positive rewards; bad actions get penalties.

| Event                              | Reward         | Why?                             |
| ---------------------------------- | -------------- | -------------------------------- |
| Clean a dirty **Kitchen** tile     | **+50**        | Kitchen is highest priority      |
| Clean a dirty **Living Room** tile | **+35**        | Medium priority                  |
| Clean a dirty **Hallway** tile     | **+20**        | Lowest priority                  |
| **All tiles cleaned!**             | **+200** bonus | Big reward for finishing the job |
| Step on an already clean tile      | **-5**         | Don't waste time on clean tiles  |
| Use "Clean" on a clean tile        | **-10**        | Pointless action                 |
| Hit a wall                         | **-5**         | Invalid move                     |
| Wait (do nothing)                  | **-3**         | Don't just stand there!          |
| Every step taken                   | **-0.1**       | Encourages efficiency            |

---

## 🧠 The Algorithms — Three Ways to Learn

This project implements three RL algorithms, from simplest to most advanced. Each one teaches the robot to clean, but they learn in different ways.

### 1. Q-Learning (Off-Policy, Tabular)

**The classic.** Q-Learning stores a big table (Q-table) that maps every (state, action) pair to an expected reward value. Think of it as a giant cheat sheet.

**How it updates:**

```
Q(s, a) ← Q(s, a) + α [ r + γ · max Q(s', a') − Q(s, a) ]
                                   ^^^^^^^^^^^
                            Uses the BEST possible next action
                            (even if it doesn't actually take it)
```

**Key idea:** Q-Learning is **off-policy** — it always assumes it will take the best action in the future, even while it's still exploring randomly. This makes it an **optimistic** learner.

**Real-world analogy:**

> Imagine learning to cook by trying recipes randomly. After each attempt, you write down "If I'm making pasta and I add salt → taste rating: 8/10". Over time, your notebook (Q-table) tells you the best action for every situation. Even while experimenting, you record what the _optimal_ move would have been.

**Best for:**

- Small, discrete state spaces (like our 2,300 states)
- When you want the theoretically optimal policy
- When you have enough memory to store the full Q-table

---

### 2. SARSA (On-Policy, Tabular)

**The cautious cousin.** SARSA also uses a Q-table, but updates differently. The name stands for **(S, A, R, S', A')** — State, Action, Reward, next State, next Action.

**How it updates:**

```
Q(s, a) ← Q(s, a) + α [ r + γ · Q(s', a') − Q(s, a) ]
                                   ^^^^^^^^
                            Uses the ACTUAL next action chosen
                            (including random exploration moves!)
```

**Key idea:** SARSA is **on-policy** — it learns the value of the policy it's _actually following_, including all the random exploratory moves. This makes it more **conservative** and **safer**.

**Real-world analogy:**

> Same cooking analogy, but this time you record what _actually_ happened: "I was making pasta, I randomly threw in chili flakes (exploration), and the result was... interesting." Your notebook reflects your real experience, not the ideal. This means you learn to avoid risky situations because your notes include all the times exploration went wrong.

**Best for:**

- When safety matters (avoid dangerous states)
- When the exploration policy matters for the value estimates
- When you want a more stable, risk-aware policy

**Example — SARSA vs Q-Learning near a wall:**

```
Imagine the robot is one step away from a wall.

Q-Learning thinks: "I'll just go forward optimally, no problem!"
    → Ignores the risk that exploration might hit the wall

SARSA thinks: "But sometimes I randomly hit the wall due to exploration..."
    → Learns to avoid being near walls (safer policy)
```

---

### 3. DQN — Deep Q-Network (Deep RL)

**The brain upgrade.** Instead of a Q-table, DQN uses a **neural network** to estimate Q-values. This lets it handle much larger state spaces and _generalize_ across similar states.

**Architecture:**

```
                   ┌─────────────────────────────────┐
                   │       Neural Network (QNet)      │
                   │                                  │
  State Features   │  Input(25) → 64 → ReLU → 64     │   Q-values
  [robot_row,      │            → ReLU → 6            │   [Q(s,a0),
   robot_col,      │                                  │    Q(s,a1),
   dirt_tile_1,    │  Policy Network + Target Network │    ...
   dirt_tile_2,    │  + Experience Replay Buffer      │    Q(s,a5)]
   ...,            │                                  │
   dirt_tile_23]   └─────────────────────────────────┘
```

**Three key innovations that make DQN work:**

| Component             | Problem It Solves                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| **Experience Replay** | Training data is correlated (sequential moves); random sampling from a buffer breaks this correlation |
| **Target Network**    | The network chases its own predictions (moving target); a frozen copy provides stable targets         |
| **Feature Vector**    | Instead of a state ID, uses meaningful features (position + dirt status of all 23 tiles)              |

**Feature vector (25 dimensions):**

```
[robot_row_normalized, robot_col_normalized, tile_1_dirty?, tile_2_dirty?, ..., tile_23_dirty?]
```

**Real-world analogy:**

> Instead of writing down every situation in a notebook (Q-table), you develop _intuition_ (neural network). You don't need to have been in the exact same situation before — if it _looks similar_ to something you've seen, you can make a good decision. A chef who's cooked 1000 meals doesn't need a recipe for every dish; they understand flavors and can improvise.

**Best for:**

- Large or continuous state spaces (e.g., robotics, game pixels)
- When you want generalization across similar states
- When tabular methods run out of memory

---

### 🤔 When Should I Use Which Algorithm?

Here are practical scenarios to help you choose:

#### Scenario 1: "I have a simple grid world with < 10,000 states"

> **Use Q-Learning.** It's simple, fast, and guaranteed to converge. The Q-table fits easily in memory, and you don't need the overhead of a neural network.

#### Scenario 2: "My robot is near dangerous cliffs / penalties"

> **Use SARSA.** Because SARSA accounts for exploration noise in its value estimates, the robot will learn to stay away from dangerous areas. Q-Learning might learn a policy that walks right along the edge (optimal but risky).

#### Scenario 3: "My state space is huge (images, continuous values)"

> **Use DQN.** When there are millions of possible states, a Q-table can't store them all. A neural network can generalize — if it has seen a _similar_ state, it can make a good guess for a new one.

#### Scenario 4: "I want the fastest training for this project"

> **Use Q-Learning.** Tabular methods are faster per step than DQN (no gradient computation). Q-Learning typically converges faster than SARSA because of its optimistic updates.

#### Scenario 5: "I'm building a self-driving car simulation"

> **Use DQN** (or more advanced algorithms). The state space (camera images, sensor data) is way too large for a table. You need function approximation.

### Quick Comparison Table

| Feature             | Q-Learning                            | SARSA                                   | DQN                                |
| ------------------- | ------------------------------------- | --------------------------------------- | ---------------------------------- |
| **Type**            | Tabular / Off-policy                  | Tabular / On-policy                     | Neural Net / Off-policy            |
| **State Space**     | Small, discrete                       | Small, discrete                         | Large, continuous                  |
| **Memory**          | Q-table (grows with states)           | Q-table (grows with states)             | Fixed (network weights)            |
| **Generalization**  | None (each state is independent)      | None                                    | Yes (similar states share weights) |
| **Update Target**   | `max Q(s', a')` (best possible)       | `Q(s', a')` (actual next action)        | Target network + replay buffer     |
| **Safety**          | Aggressive (ignores exploration risk) | Conservative (accounts for exploration) | Depends on tuning                  |
| **Training Speed**  | Fast                                  | Medium                                  | Slower (GPU helps)                 |
| **Code Complexity** | Simple                                | Simple                                  | More complex                       |
| **Best For**        | Small problems, fast prototyping      | Safety-critical tasks                   | Large/complex problems             |

---

## 📁 Project Structure

```
cleaning-robot-rl/
│
├── main.py                  # Interactive menu: train, test, compare all algorithms
├── train.py                 # Standalone Q-Learning training script
├── test.py                  # Standalone Q-Learning testing script
├── requirements.txt         # Python dependencies
│
├── env/
│   ├── __init__.py
│   └── cleaning_env.py      # Custom Gymnasium environment (house, rewards, rendering)
│
├── agent/
│   ├── __init__.py
│   ├── q_learning_agent.py  # Q-Learning agent (tabular, off-policy)
│   ├── sarsa_agent.py       # SARSA agent (tabular, on-policy)
│   └── dqn_agent.py         # DQN agent (neural network, off-policy)
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py           # Time formatting, progress bars, console utilities
│   └── plotting.py          # Matplotlib plots for training curves
│
├── models/                  # Saved trained models (auto-generated)
│   ├── q_table.pkl          # Trained Q-Learning Q-table
│   ├── sarsa_table.pkl      # Trained SARSA Q-table
│   ├── dqn_model.pth        # Trained DQN neural network weights
│   └── *_history.pkl        # Training history for comparison dashboard
│
└── plots/                   # Generated comparison charts (auto-generated)
    ├── comparison_learning_curves.png
    ├── comparison_optimal_paths.png
    ├── comparison_bars.png
    └── comparison_smart_analysis.png
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- A display (for Pygame visualization)

### Installation

1. **Clone the repository:**



2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   | Package | Purpose |
   |---------|---------|
   | `gymnasium` | RL environment framework |
   | `numpy` | Numerical operations & Q-table storage |
   | `matplotlib` | Training plots & comparison charts |
   | `pygame` | 2D visualization of the robot cleaning |
   | `torch` | Neural network for DQN agent |

### Running the Project

#### Option 1: Interactive Menu (Recommended)

```bash
python main.py
```

This opens a full interactive menu:

```
  +--------------------------------------------------+
  |                   MAIN MENU                      |
  +--------------------------------------------------+
  |  [1]  Train Q-Learning Agent                     |
  |  [2]  Train SARSA Agent                          |
  |  [3]  Train DQN Agent                            |
  |  [4]  Test  Q-Learning Agent (Pygame UI)         |
  |  [5]  Test  SARSA Agent (Pygame UI)              |
  |  [6]  Test  DQN Agent (Pygame UI)                |
  |  [7-9] Show Optimal Paths                        |
  |  [10] Compare All Algorithms (Dashboard)         |
  |  [11] Quick Train & Compare All                  |
  |  [0]  Exit                                       |
  +--------------------------------------------------+
```

#### Option 2: Quick Train & Compare (One Command)

From the menu, select **[11] Quick Train & Compare All** to:

1. Train Q-Learning, SARSA, and DQN back-to-back
2. Generate a full comparison dashboard with charts

#### Option 3: Train from Python Code

```python
from main import train

# Train Q-Learning for 5000 episodes
history = train(algo='qlearning', num_episodes=5000)

# Train SARSA for 5000 episodes
history = train(algo='sarsa', num_episodes=5000)

# Train DQN for 3000 episodes
history = train(algo='dqn', num_episodes=3000)
```

#### Option 4: Standalone Training (Q-Learning only)

```bash
python train.py
```

---

## ✨ Features

- **Three RL Algorithms** — Q-Learning, SARSA, and DQN implemented from scratch
- **Custom Gymnasium Environment** — fully featured house grid with rooms, walls, and dirt
- **Pygame Visualization** — watch the robot clean in real time with colored rooms and animations
- **Interactive Menu** — easy-to-use console interface for training, testing, and comparing
- **Comparison Dashboard** — automated 4-figure analysis with:
  - Learning curves with variance bands
  - Side-by-side optimal path visualization
  - Performance bar charts with winner badges
  - Radar chart for multi-dimensional comparison
- **Optimal Path Extraction** — see the exact route each algorithm takes
- **Random Baseline** — compare trained agents against random actions to verify learning
- **Fully Documented Code** — every file has detailed docstrings explaining the RL concepts

---

## 📈 How Training Works

Here's what happens during training (step by step):

```
Episode 1 (epsilon = 1.0 → 100% random)
├── Robot spawns on a random tile, all 23 tiles are dirty
├── Takes random actions (exploring the house)
├── Gets rewards/penalties → updates Q-values
├── Episode ends when all tiles clean OR max steps reached
└── Epsilon decays slightly

Episode 100 (epsilon ≈ 0.82 → 82% random)
├── Still mostly exploring, but starting to exploit learned values
└── Q-table is filling up with rough estimates

Episode 1000 (epsilon ≈ 0.13 → 13% random)
├── Mostly using learned policy, occasional exploration
└── Success rate improving rapidly

Episode 3000+ (epsilon ≈ 0.02 → 2% random)
├── Almost fully exploiting learned optimal policy
├── Cleans all 23 tiles consistently
└── Efficient paths through all rooms
```

### Hyperparameters

| Parameter           | Q-Learning / SARSA | DQN             | Description                  |
| ------------------- | ------------------ | --------------- | ---------------------------- |
| Learning Rate (α)   | 0.15               | 0.001           | How fast Q-values update     |
| Discount Factor (γ) | 0.99               | 0.99            | Importance of future rewards |
| Epsilon Start       | 1.0                | 1.0             | Initial exploration rate     |
| Epsilon End         | 0.02               | 0.02            | Final exploration rate       |
| Epsilon Decay       | 0.998              | 0.9987          | Decay rate per episode       |
| Default Episodes    | 5,000              | 3,000           | Recommended training length  |
| Batch Size          | —                  | 64              | DQN mini-batch size          |
| Replay Buffer       | —                  | 10,000          | DQN experience memory        |
| Target Update       | —                  | Every 100 steps | DQN target network sync      |

---

## 📊 Results & Comparison

After training all three algorithms, the comparison dashboard generates detailed analysis:

### What Gets Generated

| Chart                | What It Shows                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| **Learning Curves**  | Reward, tiles cleaned, success rate, and epsilon over episodes (with variance bands)           |
| **Optimal Paths**    | Side-by-side visualization of each algorithm's cleaning route on the house grid                |
| **Performance Bars** | Bar charts for 6 metrics (reward, tiles, success %, steps, stability, time) with winner badges |
| **Smart Analysis**   | Convergence speed, reward stability, training efficiency, and a radar/spider chart             |

### Console Output Includes

- Algorithm characteristics table (type, model size, update rule)
- Full performance summary with per-metric winners
- Strengths & weaknesses of each algorithm
- Overall winner determination

---

## 📚 Key Concepts for Beginners

New to Reinforcement Learning? Here's a quick glossary:

| Term             | Meaning                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| **Agent**        | The robot (the learner that takes actions)                              |
| **Environment**  | The house grid (everything outside the agent)                           |
| **State**        | A snapshot of the current situation (position + dirt info)              |
| **Action**       | Something the agent can do (move, clean, wait)                          |
| **Reward**       | Feedback signal — positive = good, negative = bad                       |
| **Episode**      | One complete run (start → all clean or time out)                        |
| **Policy**       | The agent's strategy (what action to take in each state)                |
| **Q-value**      | Expected total future reward for taking an action in a state            |
| **Epsilon (ε)**  | Probability of taking a random action (exploration rate)                |
| **Exploration**  | Trying random actions to discover new strategies                        |
| **Exploitation** | Using the best known action to maximize reward                          |
| **Off-policy**   | Learns optimal behavior regardless of current actions (Q-Learning, DQN) |
| **On-policy**    | Learns the value of the behavior it actually follows (SARSA)            |
| **Discount (γ)** | How much future rewards are worth compared to immediate                 |

### The Exploration-Exploitation Dilemma

This is the core challenge in RL:

```
🎰 Should I go to my favorite restaurant (exploitation)?
   Or try a new one that might be even better (exploration)?

🤖 In our project:
   - High epsilon (early training) → lots of exploration → discover the house
   - Low epsilon (late training) → mostly exploitation → clean efficiently
```

---

## 📄 License

This project is open source and available for educational purposes.

---

<p align="center">
  <b>Built with ❤️ for learning Reinforcement Learning</b><br>
  <i>From random chaos to optimal cleaning — one episode at a time</i>
</p>
