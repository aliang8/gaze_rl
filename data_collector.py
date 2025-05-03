import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from environment import TwoGoalEnv
from collections import defaultdict

class OptimalPolicy:
    """A simple optimal policy for the TwoGoalEnv with continuous actions."""
    
    def __init__(self, env):
        self.env = env
        
    def get_action(self, state, target_goal):
        """
        Get the action that moves the agent towards the target goal.
        
        Args:
            state: Current state of the agent (position)
            target_goal: Goal index (1 or 2)
        
        Returns:
            action: 2D numpy array representing direction vector (dx, dy)
        """
        # Get target goal position
        if target_goal == 1:
            goal_pos = self.env.goal1_pos
        else:
            goal_pos = self.env.goal2_pos
        
        # Calculate direction vector to goal
        direction = goal_pos - state
        
        # Normalize the direction vector to have magnitude 1.0
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            normalized_direction = direction / direction_norm
        else:
            normalized_direction = np.zeros(2)
        
        return normalized_direction

def collect_trajectories(num_episodes=1000, noise_level=0.2, noise_scale=0.5, save_path='data'):
    """
    Collect trajectories from the TwoGoalEnv using a noisy optimal policy.
    
    Args:
        num_episodes: Number of episodes to collect
        noise_level: Probability of adding noise to the action
        noise_scale: Scale of Gaussian noise to add to actions
        save_path: Directory to save the trajectory data
    
    Returns:
        trajectories: Dictionary containing all trajectory data
    """
    # Create environment
    env = TwoGoalEnv()
    
    # Create optimal policy
    policy = OptimalPolicy(env)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Store all trajectory data
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_dones = []
    all_target_goals = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Collect trajectories
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset(seed=episode)
        target_goal = info['target_goal']
        done = False
        
        # Collect trajectory
        while not done:
            # Get optimal action (direction vector)
            optimal_action = policy.get_action(state, target_goal)
            
            # Apply noise with probability noise_level
            if np.random.random() < noise_level:
                # Add Gaussian noise to the action
                noise = np.random.normal(0, noise_scale, size=2)
                action = optimal_action + noise
                
                # Re-normalize if the action exceeds bounds
                action_norm = np.linalg.norm(action)
                if action_norm > 1.0:
                    action = action / action_norm
            else:
                action = optimal_action
            
            # Take action in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Store transition
            all_states.append(state.copy())
            all_actions.append(action.copy())
            all_next_states.append(next_state.copy())
            all_rewards.append(reward)
            all_dones.append(done)
            all_target_goals.append(target_goal)
            
            # Update state
            state = next_state
        
        if (episode + 1) % 100 == 0:
            print(f"Collected {episode + 1} episodes")
    
    # Create DataFrame
    data = {
        'state_x': [s[0] for s in all_states],
        'state_y': [s[1] for s in all_states],
        'action_x': [a[0] for a in all_actions],
        'action_y': [a[1] for a in all_actions],
        'next_state_x': [s[0] for s in all_next_states],
        'next_state_y': [s[1] for s in all_next_states],
        'reward': all_rewards,
        'done': all_dones,
        'target_goal': all_target_goals
    }
    
    df = pd.DataFrame(data)
    
    # Save data
    df.to_csv(os.path.join(save_path, 'trajectories.csv'), index=False)
    
    # Create summary plot
    plot_trajectories(df, save_path)
    
    print(f"Collected {len(df)} transitions across {num_episodes} episodes")
    print(f"Data saved to {os.path.join(save_path, 'trajectories.csv')}")
    
    return df

def plot_trajectories(df, save_path):
    """
    Plot trajectories by target goal for visualization.
    
    Args:
        df: DataFrame containing trajectory data
        save_path: Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Set up environment parameters for visualization
    env = TwoGoalEnv()
    
    # Plot environment boundaries
    plt.xlim([env.x_min, env.x_max])
    plt.ylim([env.y_min, env.y_max])
    
    # Plot goals
    plt.scatter(env.goal1_pos[0], env.goal1_pos[1], color='blue', marker='o', s=200, alpha=0.6, label='Goal 1')
    plt.scatter(env.goal2_pos[0], env.goal2_pos[1], color='green', marker='o', s=200, alpha=0.6, label='Goal 2')
    
    # Plot starting position
    plt.scatter(env.start_pos[0], env.start_pos[1], color='black', marker='s', s=100, label='Start')
    
    # Group trajectories by episodes
    episode_groups = defaultdict(list)
    current_episode = 0
    
    for i in range(len(df)):
        state = np.array([df.iloc[i]['state_x'], df.iloc[i]['state_y']])
        target_goal = df.iloc[i]['target_goal']
        
        episode_groups[current_episode].append((state, target_goal))
        
        if df.iloc[i]['done']:
            current_episode += 1
    
    # Sample 50 random episodes to plot (to avoid clutter)
    if len(episode_groups) > 50:
        sample_episodes = np.random.choice(list(episode_groups.keys()), 50, replace=False)
    else:
        sample_episodes = episode_groups.keys()
    
    # Plot trajectories
    for episode in sample_episodes:
        states, target_goals = zip(*episode_groups[episode])
        target_goal = target_goals[0]  # Target goal is the same for all states in an episode
        
        # Convert states to x and y coordinates
        x_coords = [state[0] for state in states]
        y_coords = [state[1] for state in states]
        
        # Plot trajectory with color based on target goal
        color = 'blue' if target_goal == 1 else 'green'
        plt.plot(x_coords, y_coords, color=color, alpha=0.3)
    
    plt.title('Sample Trajectories by Target Goal')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(save_path, 'trajectories_plot.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Collect trajectories
    trajectories = collect_trajectories(num_episodes=1000, noise_level=0.8, noise_scale=0.5) 