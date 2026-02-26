import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent with three exploration strategies:
    1. Epsilon-greedy: Random action with probability epsilon
    2. UCB (Upper Confidence Bound): Select action based on Q + uncertainty bonus
    3. Optimistic Initialization: Start with high Q-values to encourage exploration
    """
    
    def __init__(
        self,
        n_rows=10,
        n_cols=10,
        n_actions=5,
        exploration_type='epsilon_greedy',
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        ucb_c=2.0,
        optimistic_value=20.0
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions
        self.exploration_type = exploration_type
        
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.ucb_c = ucb_c
        self.optimistic_value = optimistic_value
        
        if exploration_type == 'optimistic':
            self.q_table = np.full((n_rows, n_cols, n_actions), optimistic_value, dtype=np.float64)
        else:
            self.q_table = np.zeros((n_rows, n_cols, n_actions), dtype=np.float64)
        
        self.action_counts = np.zeros((n_rows, n_cols, n_actions), dtype=np.int64)
        self.state_visits = np.zeros((n_rows, n_cols), dtype=np.int64)
        
        self.total_steps = 0
        self.exploration_history = []
        self.action_history = []
    
    def choose_action(self, state, training=True):
        """
        Choose action based on the exploration strategy.
        
        Args:
            state: (row, col) tuple
            training: If False, always exploit (no exploration)
        
        Returns:
            action: Integer action (0-4)
        """
        x, y = state
        
        if not training:
            return int(np.argmax(self.q_table[x, y]))
        
        if self.exploration_type == 'epsilon_greedy':
            action = self._epsilon_greedy_action(x, y)
        elif self.exploration_type == 'ucb':
            action = self._ucb_action(x, y)
        elif self.exploration_type == 'optimistic':
            action = self._greedy_action(x, y)
        else:
            raise ValueError(f"Unknown exploration type: {self.exploration_type}")
        
        self.action_counts[x, y, action] += 1
        self.state_visits[x, y] += 1
        self.total_steps += 1
        self.action_history.append(action)
        
        return action
    
    def _greedy_action(self, x, y):
        """Select action with highest Q-value."""
        q_values = self.q_table[x, y]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))
    
    def _epsilon_greedy_action(self, x, y):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
            self.exploration_history.append(1)
        else:
            action = self._greedy_action(x, y)
            self.exploration_history.append(0)
        return action
    
    def _ucb_action(self, x, y):
        """
        Upper Confidence Bound action selection.
        UCB(a) = Q(s,a) + c * sqrt(log(N(total)) / N(s,a)) * decay
        
        Improved version with:
        - Global step count instead of per-state count (works better for grid-worlds)
        - Decay factor to reduce exploration over time
        """
        q_values = self.q_table[x, y].copy()
        
        total_steps = max(self.total_steps, 1)
        decay_factor = max(0.1, 1.0 - self.total_steps / 10000)
        
        for a in range(self.n_actions):
            n_action = max(self.action_counts[x, y, a], 1)
            bonus = self.ucb_c * np.sqrt(np.log(total_steps + 1) / n_action) * decay_factor
            q_values[a] += bonus
        
        return int(np.argmax(q_values))
    
    def learn(self, state, action, reward, next_state, done=False):
        """
        Update Q-table using Q-learning update rule.
        Q(s,a) <- Q(s,a) + lr * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        """
        x, y = state
        nx, ny = next_state
        
        current_q = self.q_table[x, y, action]
        
        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[nx, ny])
            target = reward + self.gamma * max_next_q
        
        self.q_table[x, y, action] += self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy exploration."""
        if self.exploration_type == 'epsilon_greedy':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_history(self):
        """Reset exploration and action history."""
        self.exploration_history = []
        self.action_history = []
    
    def get_exploration_rate(self):
        """Get current exploration rate."""
        if len(self.exploration_history) == 0:
            return 0.0
        return np.mean(self.exploration_history[-100:])
    
    def get_q_values_for_position(self, state):
        """Get Q-values for a specific position."""
        x, y = state
        return self.q_table[x, y].copy()
    
    def get_best_action_for_position(self, state):
        """Get best action for a specific position."""
        x, y = state
        return int(np.argmax(self.q_table[x, y]))
    
    def get_q_heatmap(self, action=None):
        """
        Generate a heatmap of Q-values.
        
        Args:
            action: If None, show max Q-value; otherwise show Q-value for specific action
        
        Returns:
            2D numpy array of Q-values
        """
        if action is None:
            return np.max(self.q_table, axis=2)
        else:
            return self.q_table[:, :, action]
    
    def get_visit_heatmap(self):
        """Generate a heatmap of state visit counts."""
        return self.state_visits.copy()
    
    def save_q_table(self, filepath):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)
        np.save(filepath.replace('.npy', '_counts.npy'), self.action_counts)
    
    def load_q_table(self, filepath):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)
        self.action_counts = np.load(filepath.replace('.npy', '_counts.npy'))


class ComparisonExperiment:
    """
    Class to run comparison experiments between different exploration strategies.
    """
    
    def __init__(self, env_class, agent_configs):
        self.env_class = env_class
        self.agent_configs = agent_configs
        self.results = {}
    
    def run_single_experiment(self, agent_type, n_episodes=500, max_steps=500, 
                              pattern_shift_episode=None, new_rates=None):
        """
        Run a single experiment with one agent type.
        
        Args:
            agent_type: Type of exploration ('epsilon_greedy', 'ucb', 'optimistic')
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            pattern_shift_episode: Episode at which to shift dirt patterns
            new_rates: New dirt rates after shift {'kitchen': x, 'living': y, 'hallway': z}
        
        Returns:
            Dictionary of metrics
        """
        env = self.env_class(max_steps=max_steps)
        config = self.agent_configs[agent_type]
        agent = QLearningAgent(
            n_rows=env.rows,
            n_cols=env.cols,
            n_actions=5,
            exploration_type=agent_type,
            **config
        )
        
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'dirt_collected': [],
            'dirt_remaining': [],
            'exploration_rates': [],
            'zone_visits': {'kitchen': [], 'living': [], 'hallway': []},
            'q_tables': []
        }
        
        for episode in range(n_episodes):
            if pattern_shift_episode and episode == pattern_shift_episode:
                env.set_dirt_rates(**new_rates)
            
            state = env.reset()
            episode_reward = 0
            episode_zone_visits = {'kitchen': 0, 'living': 0, 'hallway': 0}
            
            for step in range(max_steps):
                action = agent.choose_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                agent.learn(state, action, reward, next_state, done)
                
                zone = env.get_zone_name(state)
                if zone in episode_zone_visits:
                    episode_zone_visits[zone] += 1
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            agent.decay_epsilon()
            
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(info['step'])
            metrics['dirt_collected'].append(info['dirt_cleaned'])
            metrics['dirt_remaining'].append(info['dirt_remaining'])
            metrics['exploration_rates'].append(agent.get_exploration_rate())
            
            for zone in episode_zone_visits:
                metrics['zone_visits'][zone].append(episode_zone_visits[zone])
            
            if episode % 100 == 0:
                metrics['q_tables'].append(agent.q_table.copy())
        
        return metrics, agent
    
    def run_all_experiments(self, n_episodes=500, max_steps=500, 
                           pattern_shift_episode=None, new_rates=None):
        """Run experiments for all agent types."""
        for agent_type in self.agent_configs:
            print(f"Training {agent_type} agent...")
            metrics, agent = self.run_single_experiment(
                agent_type, n_episodes, max_steps, pattern_shift_episode, new_rates
            )
            self.results[agent_type] = {
                'metrics': metrics,
                'agent': agent
            }
        
        return self.results


if __name__ == "__main__":
    from environment import DirtPatternRoom
    
    print("Testing Q-Learning Agent with different exploration strategies...")
    print("=" * 60)
    
    env = DirtPatternRoom(max_steps=100)
    
    agent_configs = {
        'epsilon_greedy': {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 0.3,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        },
        'ucb': {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'ucb_c': 2.0
        },
        'optimistic': {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'optimistic_value': 20.0
        }
    }
    
    for agent_type, config in agent_configs.items():
        print(f"\n{agent_type.upper()} Agent:")
        print("-" * 40)
        
        agent = QLearningAgent(
            n_rows=env.rows,
            n_cols=env.cols,
            n_actions=5,
            exploration_type=agent_type,
            **config
        )
        
        total_reward = 0
        state = env.reset()
        
        for step in range(50):
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        
        print(f"Total reward: {total_reward:.1f}")
        print(f"Dirt cleaned: {info['dirt_cleaned']}")
        print(f"Exploration rate: {agent.get_exploration_rate():.2f}")
        if agent_type == 'epsilon_greedy':
            print(f"Current epsilon: {agent.epsilon:.3f}")
    
    print("\n" + "=" * 60)
    print("Agent test completed successfully!")
