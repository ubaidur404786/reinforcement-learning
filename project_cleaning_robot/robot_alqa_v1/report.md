# Learning Dirt Patterns: Comparing Exploration Strategies for Vacuum Cleaning Robots

**Reinforcement Learning Project Report**

---

## 1. Introduction

### 1.1 Problem Statement

Traditional vacuum cleaning robot simulations focus on cleaning a dirty environment once. However, real-world homes have areas that get dirty repeatedly at different rates. This project investigates how reinforcement learning agents can learn these "dirt patterns" and adapt their cleaning strategies accordingly.

### 1.2 Objectives

1. **Compare exploration strategies**: Evaluate ε-greedy, UCB (Upper Confidence Bound), and Optimistic Initialization in a dynamic cleaning environment
2. **Analyze pattern learning**: Determine if agents learn to prioritize high-dirt zones
3. **Measure adaptation**: Test how quickly each strategy adapts when dirt patterns change

### 1.3 Original Contribution

This project introduces:
- **Time-based dirt respawning** in different zones (unlike traditional "clean once" problems)
- **Pattern shift experiment** to test adaptation ability
- **Comprehensive comparison** of exploration strategies in a grid-world environment

---

## 2. Methodology

### 2.1 Environment Design

We designed a custom environment called `DirtPatternRoom` with the following characteristics:

| Component | Description |
|-----------|-------------|
| **Grid size** | 10×10 |
| **Actions** | 5 (Up, Down, Left, Right, Clean) |
| **Zones** | Kitchen, Living Room, Hallway |
| **Dirt model** | Time-based respawning |

#### Dirt Zone Configuration

| Zone | Tiles | Respawn Rate | Dirt Probability |
|------|-------|--------------|------------------|
| Kitchen | 9 tiles (0:3, 0:3) | Every 3 steps | 30% per tile |
| Living Room | 9 tiles (3:6, 4:7) | Every 8 steps | 20% per tile |
| Hallway | 9 tiles (7:10, 7:10) | Every 15 steps | 15% per tile |

#### Reward Structure

| Action | Reward |
|--------|--------|
| Clean dirty tile | +20 |
| Clean already-clean tile | -2 |
| Hit wall/boundary | -1 |
| Step cost | -0.05 |

### 2.2 Agent Implementation

We implemented a tabular Q-learning agent with three exploration strategies:

#### 2.2.1 ε-greedy

```
With probability ε: random action
Otherwise: greedy action (argmax Q)
```

- Initial ε: 0.3
- Decay rate: 0.995 per episode
- Minimum ε: 0.01

#### 2.2.2 UCB (Upper Confidence Bound)

```
UCB(a) = Q(s,a) + c × √(log(N) / n(a)) × decay
```

- Exploration constant c: 1.0
- Decay factor: max(0.1, 1 - steps/10000)

**Improvement**: We modified the standard UCB to use global step count instead of per-state visits, which works better in grid-world environments.

#### 2.2.3 Optimistic Initialization

- Initial Q-values: 10.0 (reduced from 20.0 for more realistic expectations)
- Learning rate: 0.2 (higher for faster convergence)

### 2.3 Hyperparameters

| Parameter | ε-greedy | UCB | Optimistic |
|-----------|----------|-----|------------|
| Learning rate | 0.15 | 0.2 | 0.2 |
| Discount factor (γ) | 0.95 | 0.95 | 0.95 |
| Exploration param | ε=0.3 | c=1.0 | init=10.0 |

### 2.4 Experiments

#### Experiment 1: Learning Efficiency
- Train for 500 episodes
- Maximum 500 steps per episode
- Track rewards, dirt collected, convergence

#### Experiment 2: Pattern Learning
- Analyze zone visit distribution
- Generate Q-value heatmaps
- Compare learned preferences vs expected

#### Experiment 3: Pattern Shift Adaptation
- Train for 200 episodes with original patterns
- **Shift**: Swap kitchen (3→15) and hallway (15→3) rates
- Continue training for 200 episodes
- Measure recovery speed

---

## 3. Results

### 3.1 Experiment 1: Learning Efficiency

| Agent | Avg Reward (last 50) | Dirt Collected | Convergence |
|-------|---------------------|----------------|-------------|
| **ε-greedy** | **-36.3** | **11.8** | 500 episodes |
| UCB | -51.2 | 0.1 | 500 episodes |
| Optimistic | -52.6 | 1.2 | 500 episodes |

**Key Finding**: ε-greedy significantly outperforms other strategies in both reward and dirt collection.

### 3.2 Experiment 2: Pattern Learning

| Agent | Kitchen Visits | Living Visits | Hallway Visits |
|-------|---------------|---------------|----------------|
| **ε-greedy** | **65.1%** ✅ | 19.6% | 15.3% |
| UCB | 25.3% ❌ | 40.3% | 34.4% |
| Optimistic | 31.5% ❌ | 38.6% | 29.9% |

**Expected distribution**: Kitchen 63.5%, Living 23.8%, Hallway 12.7%

**Key Finding**: Only ε-greedy learned to prioritize the kitchen (high-dirt zone). UCB and Optimistic agents distributed visits more uniformly.

### 3.3 Experiment 3: Pattern Shift Adaptation

| Agent | Pre-shift Avg | Post-shift Min | Episodes to Recover |
|-------|--------------|----------------|---------------------|
| ε-greedy | +164.7 | -75.0 | 200 |
| UCB | -60.3 | -75.0 | 200 |
| **Optimistic** | -63.8 | -117.0 | **103** ⚡ |

**Key Finding**: Optimistic initialization adapts fastest after pattern shift, recovering in ~100 episodes vs 200 for others.

---

## 4. Discussion

### 4.1 Why ε-greedy Performs Best

1. **Simple and effective**: The decaying ε allows initial exploration followed by exploitation
2. **Pattern learning**: Learns to prioritize kitchen (65% of visits vs expected 63.5%)
3. **Stable convergence**: Consistent performance improvement over training

### 4.2 Why UCB Struggles

1. **Designed for bandits**: UCB formula assumes single state, not grid-world navigation
2. **Over-exploration**: Even with reduced c=1.0, explores too much
3. **Scale issues**: 100 states × 5 actions = 500 state-action pairs, too many for UCB to handle efficiently

### 4.3 Why Optimistic Initialization Adapts Fastest

1. **Reset mechanism**: High initial Q-values encourage visiting all states after pattern shift
2. **Rapid learning**: Higher learning rate (0.2) allows quick policy updates
3. **Built-in exploration**: No need for explicit exploration parameter

### 4.4 Trade-offs

| Strategy | Pattern Learning | Adaptation Speed | Stability |
|----------|-----------------|------------------|-----------|
| ε-greedy | ✅ Best | ❌ Slow | ✅ High |
| UCB | ❌ Poor | ❌ Slow | ❌ Low |
| Optimistic | ❌ Moderate | ✅ Best | ⚠️ Medium |

---

## 5. Conclusions

### 5.1 Main Findings

1. **ε-greedy is best for stable environments** where the task is to learn and exploit known patterns
2. **Optimistic initialization excels at adaptation** when environment dynamics change
3. **UCB is not suitable for grid-world navigation** tasks with many states

### 5.2 Practical Recommendations

| Scenario | Recommended Strategy |
|----------|---------------------|
| Home with stable dirt patterns | ε-greedy |
| Environment with changing patterns | Optimistic initialization |
| Unknown environment type | ε-greedy with higher initial ε |

### 5.3 Limitations

1. **Grid size**: Only tested on 10×10 grid
2. **Agent type**: Only tabular Q-learning tested
3. **Dirt model**: Simple time-based respawning
4. **Single robot**: No multi-agent scenarios

### 5.4 Future Work

1. Test on larger grids with function approximation (DQN)
2. Add battery/charging constraints
3. Implement multi-robot coordination
4. Test with different dirt spawning models
5. Compare with policy gradient methods

---

## 6. References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

2. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2), 235-256.

3. Even-Dar, E., & Mansour, Y. (2003). Learning rates for Q-learning. *Journal of Machine Learning Research*, 5, 1-25.

4. Littman, M. L. (1996). Algorithms for sequential decision making. PhD thesis, Brown University.

---

## Appendix A: Code Structure

```
project_cleaning_robot/
├── environment.py      # DirtPatternRoom class
├── agents.py           # QLearningAgent with 3 exploration modes
├── utils.py            # Visualization functions
├── project_notebook.ipynb  # Main Jupyter notebook
├── results/            # Generated plots
│   ├── zones_layout.png
│   ├── exp1_comparison.png
│   ├── exp2_zone_visits.png
│   ├── exp2_q_heatmap_*.png
│   └── exp3_pattern_shift.png
└── report.md           # This report
```

## Appendix B: Hyperparameter Sensitivity

We tested various hyperparameters and found:

| Parameter | Effect |
|-----------|--------|
| Learning rate 0.1-0.2 | Higher = faster convergence but less stable |
| Discount factor 0.9-0.99 | Higher = more long-term planning |
| ε initial 0.2-0.5 | Higher = more exploration early |
| UCB c 0.5-2.0 | Higher = more exploration |
| Optimistic init 5-20 | Higher = more initial exploration |

---

*Report generated for Reinforcement Learning course project.*
