import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class DirtPatternRoom:
    """
    A vacuum cleaning environment where dirt respawns over time in different zones.
    The robot must learn which zones get dirty most often and prioritize them.
    """
    
    def __init__(self, rows=10, cols=10, max_steps=500):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        
        self.start_pos = (0, 0)
        
        self.zone_kitchen = {(r, c) for r in range(3) for c in range(3)}
        self.zone_living = {(r, c) for r in range(3, 6) for c in range(4, 7)}
        self.zone_hallway = {(r, c) for r in range(7, 10) for c in range(7, 10)}
        
        self.dirt_respawn_rates = {
            'kitchen': 3,
            'living': 8,
            'hallway': 15
        }
        
        self.zone_counters = {
            'kitchen': 0,
            'living': 0,
            'hallway': 0
        }
        
        self.dirt_grid = np.zeros((rows, cols), dtype=np.int8)
        self.robot_pos = self.start_pos
        self.step_count = 0
        self.total_dirt_spawned = 0
        self.total_dirt_cleaned = 0
        
        self.action_names = ['Up', 'Down', 'Left', 'Right', 'Clean']
        self.reset()
    
    def reset(self):
        """Reset environment for new episode."""
        self.robot_pos = self.start_pos
        self.dirt_grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.step_count = 0
        self.total_dirt_spawned = 0
        self.total_dirt_cleaned = 0
        
        self.zone_counters = {
            'kitchen': 0,
            'living': 0,
            'hallway': 0
        }
        
        self._spawn_initial_dirt()
        
        return self._get_state()
    
    def _spawn_initial_dirt(self):
        """Spawn some initial dirt in each zone."""
        for pos in self.zone_kitchen:
            if np.random.random() < 0.5:
                self.dirt_grid[pos] = 1
                self.total_dirt_spawned += 1
        for pos in self.zone_living:
            if np.random.random() < 0.3:
                self.dirt_grid[pos] = 1
                self.total_dirt_spawned += 1
        for pos in self.zone_hallway:
            if np.random.random() < 0.2:
                self.dirt_grid[pos] = 1
                self.total_dirt_spawned += 1
    
    def _spawn_dirt(self):
        """Spawn dirt based on time-based respawn rates."""
        self.zone_counters['kitchen'] += 1
        self.zone_counters['living'] += 1
        self.zone_counters['hallway'] += 1
        
        if self.zone_counters['kitchen'] >= self.dirt_respawn_rates['kitchen']:
            for pos in self.zone_kitchen:
                if np.random.random() < 0.3:
                    self.dirt_grid[pos] = 1
                    self.total_dirt_spawned += 1
            self.zone_counters['kitchen'] = 0
        
        if self.zone_counters['living'] >= self.dirt_respawn_rates['living']:
            for pos in self.zone_living:
                if np.random.random() < 0.2:
                    self.dirt_grid[pos] = 1
                    self.total_dirt_spawned += 1
            self.zone_counters['living'] = 0
        
        if self.zone_counters['hallway'] >= self.dirt_respawn_rates['hallway']:
            for pos in self.zone_hallway:
                if np.random.random() < 0.15:
                    self.dirt_grid[pos] = 1
                    self.total_dirt_spawned += 1
            self.zone_counters['hallway'] = 0
    
    def _get_state(self):
        """Get current state representation."""
        return (self.robot_pos[0], self.robot_pos[1])
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).
        
        Actions:
            0: Up (decrease row)
            1: Down (increase row)
            2: Left (decrease col)
            3: Right (increase col)
            4: Clean current tile
        """
        self.step_count += 1
        reward = 0.0
        done = False
        
        x, y = self.robot_pos
        
        if action == 0:
            new_x, new_y = x - 1, y
        elif action == 1:
            new_x, new_y = x + 1, y
        elif action == 2:
            new_x, new_y = x, y - 1
        elif action == 3:
            new_x, new_y = x, y + 1
        elif action == 4:
            new_x, new_y = x, y
            
            if self.dirt_grid[x, y] == 1:
                reward = 20.0
                self.dirt_grid[x, y] = 0
                self.total_dirt_cleaned += 1
            else:
                reward = -2.0
        else:
            raise ValueError(f"Invalid action: {action}")
        
        if action in [0, 1, 2, 3]:
            if (0 <= new_x < self.rows and 0 <= new_y < self.cols):
                self.robot_pos = (new_x, new_y)
            else:
                reward = -1.0
        
        reward -= 0.05
        
        self._spawn_dirt()
        
        if self.step_count >= self.max_steps:
            done = True
        
        info = {
            'dirt_cleaned': self.total_dirt_cleaned,
            'dirt_remaining': np.sum(self.dirt_grid),
            'dirt_spawned': self.total_dirt_spawned,
            'step': self.step_count
        }
        
        return self._get_state(), reward, done, info
    
    def get_zone_name(self, pos):
        """Get the zone name for a position."""
        if pos in self.zone_kitchen:
            return 'kitchen'
        elif pos in self.zone_living:
            return 'living'
        elif pos in self.zone_hallway:
            return 'hallway'
        else:
            return 'neutral'
    
    def set_dirt_rates(self, kitchen=3, living=8, hallway=15):
        """Update dirt respawn rates (for pattern shift experiment)."""
        self.dirt_respawn_rates['kitchen'] = kitchen
        self.dirt_respawn_rates['living'] = living
        self.dirt_respawn_rates['hallway'] = hallway
    
    def render(self, mode='human'):
        """Render the environment as a text grid."""
        symbols = {
            'robot': 'R',
            'dirt': 'D',
            'clean': '.',
            'zone_kitchen': 'K',
            'zone_living': 'L',
            'zone_hallway': 'H'
        }
        
        grid_str = f"Step: {self.step_count}\n"
        grid_str += f"Dirt Cleaned: {self.total_dirt_cleaned} | Remaining: {np.sum(self.dirt_grid)}\n"
        grid_str += "-" * (self.cols * 2 + 1) + "\n"
        
        for r in range(self.rows):
            row_str = "|"
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.robot_pos:
                    row_str += "R|"
                elif self.dirt_grid[r, c] == 1:
                    row_str += "D|"
                elif pos in self.zone_kitchen:
                    row_str += "k|"
                elif pos in self.zone_living:
                    row_str += "l|"
                elif pos in self.zone_hallway:
                    row_str += "h|"
                else:
                    row_str += ".|"
            grid_str += row_str + "\n"
        
        grid_str += "-" * (self.cols * 2 + 1)
        print(grid_str)
    
    def visualize_zones(self, ax=None, figsize=(8, 8)):
        """Visualize the dirt zones."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        ax.set_aspect('equal')
        
        zone_colors = {
            'kitchen': '#FFE4E1',
            'living': '#E6E6FA',
            'hallway': '#F0FFF0'
        }
        
        for pos in self.zone_kitchen:
            rect = Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, 
                            facecolor=zone_colors['kitchen'], alpha=0.5)
            ax.add_patch(rect)
        
        for pos in self.zone_living:
            rect = Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1,
                            facecolor=zone_colors['living'], alpha=0.5)
            ax.add_patch(rect)
        
        for pos in self.zone_hallway:
            rect = Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1,
                            facecolor=zone_colors['hallway'], alpha=0.5)
            ax.add_patch(rect)
        
        ax.plot(self.start_pos[1], self.start_pos[0], 'g^', markersize=15, label='Start')
        
        ax.legend(loc='upper right')
        ax.set_title('Dirt Zones Layout\nK=Kitchen (fast), L=Living (medium), H=Hallway (slow)', 
                    fontsize=12)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.grid(True, alpha=0.3)
        
        return ax


if __name__ == "__main__":
    env = DirtPatternRoom()
    print("Testing DirtPatternRoom environment...")
    print("=" * 50)
    
    env.visualize_zones()
    plt.savefig('zones_layout.png', dpi=150, bbox_inches='tight')
    print("Zone layout saved to 'zones_layout.png'")
    
    print("\nInitial state:")
    env.render()
    
    state = env.reset()
    total_reward = 0
    
    for step in range(20):
        action = np.random.randint(0, 5)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"\nStep {step}: Action={env.action_names[action]}, Reward={reward:.1f}")
            env.render()
        
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.1f}")
    print(f"Dirt cleaned: {info['dirt_cleaned']}")
