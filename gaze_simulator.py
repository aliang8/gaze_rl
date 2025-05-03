import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import os
from environment import TwoGoalEnv

class GazeSimulator:
    """
    Simulates human gaze towards the target goal.
    
    This class provides methods to generate gaze data that would
    represent a human looking at the target goal, with some noise
    and variability to make it realistic.
    """
    
    def __init__(self, env, gaze_noise_level=0.2):
        """
        Initialize the gaze simulator.
        
        Args:
            env: TwoGoalEnv instance
            gaze_noise_level: Standard deviation of Gaussian noise added to gaze
        """
        self.env = env
        self.gaze_noise_level = gaze_noise_level
    
    def generate_gaze(self, state, target_goal):
        """
        Generate simulated gaze data.
        
        Args:
            state: Current agent state (position)
            target_goal: Index of the target goal (1 or 2)
        
        Returns:
            gaze: 2D numpy array representing the gaze position
        """
        # Get goal position based on target goal
        if target_goal == 1:
            goal_pos = self.env.goal1_pos
        else:
            goal_pos = self.env.goal2_pos
        
        # Generate gaze with some noise
        gaze_noise = np.random.normal(0, self.gaze_noise_level, size=2)
        gaze = goal_pos + gaze_noise
        
        # Ensure gaze stays within environment bounds
        gaze[0] = np.clip(gaze[0], self.env.x_min, self.env.x_max)
        gaze[1] = np.clip(gaze[1], self.env.y_min, self.env.y_max)
        
        return gaze

def add_gaze_to_trajectories(
    trajectory_path='data/trajectories.csv',
    output_path='data/trajectories_with_gaze.csv',
    gaze_noise_level=0.2
):
    """
    Add simulated gaze information to trajectory data.
    
    Args:
        trajectory_path: Path to the trajectory data CSV file
        output_path: Path to save the trajectory data with gaze
        gaze_noise_level: Standard deviation of Gaussian noise added to gaze
    
    Returns:
        df: DataFrame containing trajectory data with gaze information
    """
    # Load trajectory data
    df = pd.read_csv(trajectory_path)
    
    # Create environment and gaze simulator
    env = TwoGoalEnv()
    gaze_simulator = GazeSimulator(env, gaze_noise_level=gaze_noise_level)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate gaze data for each trajectory
    gaze_x = []
    gaze_y = []
    
    for i in range(len(df)):
        state = np.array([df.iloc[i]['state_x'], df.iloc[i]['state_y']])
        target_goal = df.iloc[i]['target_goal']
        
        # Generate gaze
        gaze = gaze_simulator.generate_gaze(state, target_goal)
        
        gaze_x.append(gaze[0])
        gaze_y.append(gaze[1])
    
    # Add gaze data to DataFrame
    df['gaze_x'] = gaze_x
    df['gaze_y'] = gaze_y
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Added gaze information to trajectories: {output_path}")
    
    return df

def visualize_trajectories_with_gaze(df, save_path='data/trajectories_with_gaze.png'):
    """
    Visualize trajectories and gaze data.
    
    Args:
        df: DataFrame containing trajectory data with gaze
        save_path: Path to save the visualization
    """
    # Create a new figure
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
    
    # Sample trajectories to visualize (at most 5)
    unique_episodes = df['done'].cumsum().unique()
    if len(unique_episodes) > 5:
        sample_episodes = np.random.choice(unique_episodes, 5, replace=False)
    else:
        sample_episodes = unique_episodes
    
    # Plot each trajectory
    episode_idx = df['done'].cumsum()
    for episode in sample_episodes:
        episode_data = df[episode_idx == episode]
        
        # Get target goal for this episode
        target_goal = episode_data.iloc[0]['target_goal']
        
        # Plot agent trajectory
        plt.plot(
            episode_data['state_x'], 
            episode_data['state_y'], 
            color='red', 
            linestyle='-', 
            alpha=0.6,
            label='Agent Trajectory' if episode == sample_episodes[0] else ""
        )
        
        # Plot gaze data
        plt.scatter(
            episode_data['gaze_x'], 
            episode_data['gaze_y'], 
            color='purple' if target_goal == 1 else 'orange',
            marker='x', 
            alpha=0.4,
            s=30,
            label=f'Gaze (Goal {target_goal})' if episode == sample_episodes[0] else ""
        )
    
    plt.title('Sample Trajectories with Gaze')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    
    # Save visualization
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # Add gaze to trajectories
    df = add_gaze_to_trajectories(
        trajectory_path='data/trajectories.csv',
        output_path='data/trajectories_with_gaze.csv',
        gaze_noise_level=0.2
    )
    
    # Visualize trajectories with gaze
    visualize_trajectories_with_gaze(df, save_path='data/trajectories_with_gaze.png') 