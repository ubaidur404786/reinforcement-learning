import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def plot_learning_curves(results, metric='episode_rewards', window=20, figsize=(12, 6), ax=None):
    """
    Plot learning curves for all agent types.
    
    Args:
        results: Dictionary from ComparisonExperiment.run_all_experiments()
        metric: Which metric to plot ('episode_rewards', 'dirt_collected', etc.)
        window: Smoothing window size
        figsize: Figure size
        ax: Existing axes to plot on (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    colors = {
        'epsilon_greedy': '#2ecc71',
        'ucb': '#3498db',
        'optimistic': '#e74c3c'
    }
    
    for agent_type, data in results.items():
        values = data['metrics'][metric]
        
        smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
        
        ax.plot(smoothed, label=agent_type.replace('_', ' ').title(), 
               color=colors.get(agent_type, 'gray'), linewidth=2)
        
        ax.fill_between(range(len(smoothed)), 
                       smoothed - np.std(values[:len(smoothed)]),
                       smoothed + np.std(values[:len(smoothed)]),
                       color=colors.get(agent_type, 'gray'), alpha=0.2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Learning Curves: {metric.replace("_", " ").title()}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_zone_visits(results, figsize=(14, 5), axes=None):
    """
    Plot zone visit distribution for each agent.
    
    Args:
        results: Dictionary from ComparisonExperiment.run_all_experiments()
        figsize: Figure size
        axes: Existing axes to plot on (optional)
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].figure
    
    zones = ['kitchen', 'living', 'hallway']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, (agent_type, data) in enumerate(results.items()):
        zone_visits = data['metrics']['zone_visits']
        
        avg_visits = [np.mean(zone_visits[z][-50:]) for z in zones]
        
        bars = axes[idx].bar(zones, avg_visits, color=colors, alpha=0.7, edgecolor='black')
        axes[idx].set_title(agent_type.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_ylabel('Average Visits per Episode', fontsize=10)
        axes[idx].set_xlabel('Zone', fontsize=10)
        
        for bar, val in zip(bars, avg_visits):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Zone Visit Distribution (Last 50 Episodes)', fontsize=14)
    plt.tight_layout()
    return fig, axes


def plot_q_heatmap(agent, env, action=None, figsize=(10, 8)):
    """
    Plot Q-value heatmap for an agent.
    
    Args:
        agent: Trained QLearningAgent
        env: DirtPatternRoom environment
        action: Specific action (None for max Q-value)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    q_values = agent.get_q_heatmap(action)
    
    cmap = LinearSegmentedColormap.from_list('custom', 
                                              ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f'])
    
    im = ax.imshow(q_values, cmap=cmap, aspect='auto')
    
    for pos in env.zone_kitchen:
        rect = mpatches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    for pos in env.zone_living:
        rect = mpatches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    
    for pos in env.zone_hallway:
        rect = mpatches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)
    
    ax.plot(env.start_pos[1], env.start_pos[0], 'w^', markersize=15, 
           markeredgecolor='black', label='Start')
    
    plt.colorbar(im, ax=ax, label='Q-value')
    
    if action is None:
        title = 'Max Q-value Heatmap'
    else:
        action_names = ['Up', 'Down', 'Left', 'Right', 'Clean']
        title = f'Q-value Heatmap for Action: {action_names[action]}'
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='red', label='Kitchen'),
        mpatches.Patch(facecolor='none', edgecolor='blue', label='Living Room'),
        mpatches.Patch(facecolor='none', edgecolor='green', label='Hallway')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax


def plot_visit_heatmap(agent, env, figsize=(10, 8)):
    """
    Plot state visit heatmap.
    
    Args:
        agent: Trained QLearningAgent
        env: DirtPatternRoom environment
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    visits = agent.get_visit_heatmap()
    
    cmap = LinearSegmentedColormap.from_list('visits',
                                              ['#ffffff', '#f39c12', '#c0392b'])
    
    im = ax.imshow(visits, cmap=cmap, aspect='auto')
    
    for pos in env.zone_kitchen:
        rect = mpatches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    for pos in env.zone_living:
        rect = mpatches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    
    for pos in env.zone_hallway:
        rect = mpatches.Rectangle((pos[1]-0.5, pos[0]-0.5), 1, 1,
                                   fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)
    
    plt.colorbar(im, ax=ax, label='Visit Count')
    
    ax.set_title('State Visit Frequency', fontsize=14)
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='red', label='Kitchen'),
        mpatches.Patch(facecolor='none', edgecolor='blue', label='Living Room'),
        mpatches.Patch(facecolor='none', edgecolor='green', label='Hallway')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, ax


def plot_comparison_bar(results, metric='episode_rewards', last_n=50, figsize=(10, 6)):
    """
    Plot bar chart comparing final performance across agents.
    
    Args:
        results: Dictionary from ComparisonExperiment.run_all_experiments()
        metric: Which metric to compare
        last_n: Number of episodes to average over
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    agent_types = list(results.keys())
    means = []
    stds = []
    
    for agent_type in agent_types:
        values = results[agent_type]['metrics'][metric][-last_n:]
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(range(len(agent_types)), means, yerr=stds, 
                 color=colors, alpha=0.7, edgecolor='black', capsize=5)
    
    ax.set_xticks(range(len(agent_types)))
    ax.set_xticklabels([t.replace('_', ' ').title() for t in agent_types], fontsize=11)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Final Performance Comparison (Last {last_n} Episodes)', fontsize=14)
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_pattern_shift(results, shift_episode, figsize=(14, 6)):
    """
    Plot adaptation to pattern shift.
    
    Args:
        results: Dictionary from ComparisonExperiment.run_all_experiments()
        shift_episode: Episode at which pattern shift occurred
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {
        'epsilon_greedy': '#2ecc71',
        'ucb': '#3498db',
        'optimistic': '#e74c3c'
    }
    
    window = 10
    
    for agent_type, data in results.items():
        rewards = data['metrics']['episode_rewards']
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        ax.plot(smoothed, label=agent_type.replace('_', ' ').title(),
               color=colors.get(agent_type, 'gray'), linewidth=2)
    
    ax.axvline(x=shift_episode, color='red', linestyle='--', linewidth=2, 
              label='Pattern Shift')
    
    ax.fill_between([shift_episode-5, shift_episode+5], 
                   ax.get_ylim()[0], ax.get_ylim()[1],
                   color='red', alpha=0.1)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward (Smoothed)', fontsize=12)
    ax.set_title('Adaptation to Pattern Shift', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_adaptation_speed(results, shift_episode, recovery_threshold=0.8, figsize=(10, 6)):
    """
    Analyze and plot adaptation speed after pattern shift.
    
    Args:
        results: Dictionary from ComparisonExperiment.run_all_experiments()
        shift_episode: Episode at which pattern shift occurred
        recovery_threshold: Fraction of pre-shift performance to consider "recovered"
        figsize: Figure size
    """
    adaptation_data = []
    
    for agent_type, data in results.items():
        rewards = data['metrics']['episode_rewards']
        
        pre_shift_avg = np.mean(rewards[shift_episode-50:shift_episode])
        target = pre_shift_avg * recovery_threshold
        
        post_shift_rewards = rewards[shift_episode:]
        recovery_episode = None
        
        for i in range(len(post_shift_rewards)):
            window_avg = np.mean(post_shift_rewards[max(0,i-10):i+10])
            if window_avg >= target:
                recovery_episode = i
                break
        
        adaptation_data.append({
            'agent': agent_type,
            'recovery_episode': recovery_episode if recovery_episode else len(post_shift_rewards),
            'pre_shift_avg': pre_shift_avg,
            'post_shift_min': np.min(post_shift_rewards[:20]) if len(post_shift_rewards) >= 20 else np.min(post_shift_rewards)
        })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    agents = [d['agent'].replace('_', ' ').title() for d in adaptation_data]
    recovery_times = [d['recovery_episode'] for d in adaptation_data]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(agents, recovery_times, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Episodes to Recover', fontsize=12)
    ax.set_title(f'Adaptation Speed (Recovery to {recovery_threshold*100:.0f}% of Pre-shift Performance)', 
                fontsize=12)
    
    for bar, val in zip(bars, recovery_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax, adaptation_data


def create_summary_table(results, last_n=50):
    """
    Create a summary table of all metrics.
    
    Args:
        results: Dictionary from ComparisonExperiment.run_all_experiments()
        last_n: Number of episodes to average over
    
    Returns:
        Dictionary of summary statistics
    """
    summary = {}
    
    for agent_type, data in results.items():
        metrics = data['metrics']
        
        summary[agent_type] = {
            'avg_reward': np.mean(metrics['episode_rewards'][-last_n:]),
            'std_reward': np.std(metrics['episode_rewards'][-last_n:]),
            'avg_dirt_collected': np.mean(metrics['dirt_collected'][-last_n:]),
            'avg_episode_length': np.mean(metrics['episode_lengths'][-last_n:]),
            'convergence_episode': _find_convergence(metrics['episode_rewards']),
            'kitchen_visits': np.mean(metrics['zone_visits']['kitchen'][-last_n:]),
            'living_visits': np.mean(metrics['zone_visits']['living'][-last_n:]),
            'hallway_visits': np.mean(metrics['zone_visits']['hallway'][-last_n:])
        }
    
    return summary


def _find_convergence(rewards, threshold=0.95, window=50):
    """
    Find the episode where the agent converged (reached threshold of max performance).
    """
    max_reward = np.max(rewards)
    target = max_reward * threshold
    
    for i in range(window, len(rewards)):
        if np.mean(rewards[i-window:i]) >= target:
            return i - window
    
    return len(rewards)


def print_summary(summary):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    header = f"{'Agent':<20} {'Avg Reward':>12} {'Dirt Collected':>15} {'Convergence':>12}"
    print(header)
    print("-" * 80)
    
    for agent_type, stats in summary.items():
        row = f"{agent_type.replace('_', ' ').title():<20} "
        row += f"{stats['avg_reward']:>12.1f} "
        row += f"{stats['avg_dirt_collected']:>15.1f} "
        row += f"{stats['convergence_episode']:>12}"
        print(row)
    
    print("=" * 80)
    print("\nZone Visits (avg per episode, last 50 episodes):")
    print("-" * 80)
    
    for agent_type, stats in summary.items():
        print(f"\n{agent_type.replace('_', ' ').title()}:")
        print(f"  Kitchen: {stats['kitchen_visits']:.1f}")
        print(f"  Living:  {stats['living_visits']:.1f}")
        print(f"  Hallway: {stats['hallway_visits']:.1f}")


if __name__ == "__main__":
    print("Testing visualization functions...")
    print("=" * 50)
    
    np.random.seed(42)
    
    mock_results = {
        'epsilon_greedy': {
            'metrics': {
                'episode_rewards': list(np.cumsum(np.random.randn(500) * 10 + 50)),
                'dirt_collected': list(np.random.randint(5, 20, 500)),
                'episode_lengths': list(np.random.randint(100, 500, 500)),
                'zone_visits': {
                    'kitchen': list(np.random.randint(10, 30, 500)),
                    'living': list(np.random.randint(5, 20, 500)),
                    'hallway': list(np.random.randint(2, 10, 500))
                }
            }
        },
        'ucb': {
            'metrics': {
                'episode_rewards': list(np.cumsum(np.random.randn(500) * 8 + 55)),
                'dirt_collected': list(np.random.randint(8, 25, 500)),
                'episode_lengths': list(np.random.randint(100, 450, 500)),
                'zone_visits': {
                    'kitchen': list(np.random.randint(15, 35, 500)),
                    'living': list(np.random.randint(8, 25, 500)),
                    'hallway': list(np.random.randint(3, 12, 500))
                }
            }
        },
        'optimistic': {
            'metrics': {
                'episode_rewards': list(np.cumsum(np.random.randn(500) * 12 + 45)),
                'dirt_collected': list(np.random.randint(5, 18, 500)),
                'episode_lengths': list(np.random.randint(150, 500, 500)),
                'zone_visits': {
                    'kitchen': list(np.random.randint(8, 25, 500)),
                    'living': list(np.random.randint(5, 18, 500)),
                    'hallway': list(np.random.randint(2, 8, 500))
                }
            }
        }
    }
    
    print("\n1. Testing plot_learning_curves...")
    fig1, ax1 = plot_learning_curves(mock_results)
    plt.savefig('test_learning_curves.png', dpi=100, bbox_inches='tight')
    print("   Saved: test_learning_curves.png")
    
    print("\n2. Testing plot_comparison_bar...")
    fig2, ax2 = plot_comparison_bar(mock_results)
    plt.savefig('test_comparison.png', dpi=100, bbox_inches='tight')
    print("   Saved: test_comparison.png")
    
    print("\n3. Testing create_summary_table...")
    summary = create_summary_table(mock_results)
    print_summary(summary)
    
    print("\n" + "=" * 50)
    print("Visualization tests completed!")
