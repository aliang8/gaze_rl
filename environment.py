import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import gymnasium as gym
from gymnasium import spaces

class TwoGoalEnv(gym.Env):
    """
    A simple 2D environment with a point agent and two goals.
    The agent needs to navigate to one of the two goals.
    The action space is continuous, representing direction and magnitude.
    """
    
    def __init__(self, goal_distance=0.8, max_action_magnitude=0.1):
        super(TwoGoalEnv, self).__init__()
        
        # Environment bounds
        self.x_min, self.x_max = -1.0, 1.0
        self.y_min, self.y_max = -1.0, 1.0
        
        # Goals positioned symmetrically on the x-axis
        self.goal_distance = goal_distance
        self.goal1_pos = np.array([-goal_distance, 0.0])
        self.goal2_pos = np.array([goal_distance, 0.0])
        
        # Target goal (1 or 2)
        self.target_goal = None
        
        # Goal threshold (distance within which goal is considered reached)
        self.goal_threshold = 0.1
        
        # Agent starting position
        self.start_pos = np.array([0.0, -0.8])
        self.agent_pos = self.start_pos.copy()
        
        # Movement parameters
        self.max_action_magnitude = max_action_magnitude
        
        # Action space: Continuous 2D vector (dx, dy)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: Agent position (x, y)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min]),
            high=np.array([self.x_max, self.y_max]),
            dtype=np.float32
        )
        
        # Maximum episode steps
        self.max_steps = 100
        self.current_step = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position
        self.agent_pos = self.start_pos.copy()
        
        # Reset step counter
        self.current_step = 0
        
        # Randomly select the target goal (1 or 2)
        self.target_goal = self.np_random.integers(1, 3)  # 1 or 2
        
        # Return initial observation and info
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Update step counter
        self.current_step += 1
        
        # Ensure action is within bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to max_action_magnitude
        delta = action * self.max_action_magnitude
        
        # Update agent position
        self.agent_pos += delta
        
        # Clip position to ensure it stays within bounds
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.x_min, self.x_max)
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.y_min, self.y_max)
        
        # Calculate distances to both goals
        dist_to_goal1 = np.linalg.norm(self.agent_pos - self.goal1_pos)
        dist_to_goal2 = np.linalg.norm(self.agent_pos - self.goal2_pos)
        
        # Check if agent reached the target goal
        if (self.target_goal == 1 and dist_to_goal1 < self.goal_threshold) or \
           (self.target_goal == 2 and dist_to_goal2 < self.goal_threshold):
            reward = 1.0
            done = True
        # Check if agent reached the wrong goal
        elif (self.target_goal == 1 and dist_to_goal2 < self.goal_threshold) or \
             (self.target_goal == 2 and dist_to_goal1 < self.goal_threshold):
            reward = -1.0
            done = True
        else:
            # Small negative reward to encourage reaching the goal quickly
            reward = -0.01
            done = False
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True
        
        # Create info dictionary
        info = self._get_info()
        
        return self._get_obs(), reward, done, False, info
    
    def _get_obs(self):
        # Return agent position as observation
        return np.array(self.agent_pos, dtype=np.float32)
    
    def _get_info(self):
        # Calculate distances to both goals
        dist_to_goal1 = np.linalg.norm(self.agent_pos - self.goal1_pos)
        dist_to_goal2 = np.linalg.norm(self.agent_pos - self.goal2_pos)
        
        return {
            "distance_to_goal1": dist_to_goal1,
            "distance_to_goal2": dist_to_goal2,
            "target_goal": self.target_goal
        }
    
    def render(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Set plot limits
        ax.set_xlim([self.x_min, self.x_max])
        ax.set_ylim([self.y_min, self.y_max])
        
        # Plot goals
        goal1_color = 'green' if self.target_goal == 1 else 'blue'
        goal2_color = 'green' if self.target_goal == 2 else 'blue'
        
        goal1 = Circle(self.goal1_pos, self.goal_threshold, color=goal1_color, alpha=0.6)
        goal2 = Circle(self.goal2_pos, self.goal_threshold, color=goal2_color, alpha=0.6)
        
        ax.add_patch(goal1)
        ax.add_patch(goal2)
        
        # Plot agent
        agent = Circle(self.agent_pos, 0.05, color='red')
        ax.add_patch(agent)
        
        # Add labels
        plt.title(f"Target Goal: {self.target_goal}")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        
        plt.tight_layout()
        plt.show()
        
    def render_on_axes(self, ax):
        """
        Render the environment on the provided axes.
        
        Args:
            ax: matplotlib axes on which to render
        """
        # Set plot limits
        ax.set_xlim([self.x_min, self.x_max])
        ax.set_ylim([self.y_min, self.y_max])
        
        # Plot goals
        goal1_color = 'green' if self.target_goal == 1 else 'blue'
        goal2_color = 'green' if self.target_goal == 2 else 'blue'
        
        goal1 = Circle(self.goal1_pos, self.goal_threshold, color=goal1_color, alpha=0.6)
        goal2 = Circle(self.goal2_pos, self.goal_threshold, color=goal2_color, alpha=0.6)
        
        ax.add_patch(goal1)
        ax.add_patch(goal2)
        
        # Plot agent
        agent = Circle(self.agent_pos, 0.05, color='red')
        ax.add_patch(agent)
        
        # Add grid
        ax.grid(True)

if __name__ == "__main__":
    # Test the environment
    env = TwoGoalEnv()
    obs, info = env.reset()
    
    print(f"Initial observation: {obs}")
    print(f"Target goal: {info['target_goal']}")
    
    env.render()
    
    # Try some random continuous actions
    for _ in range(5):
        action = env.action_space.sample()
        print(f"Taking action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        print(f"New position: {obs}")
        env.render()
        
        if done:
            break 