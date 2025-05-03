import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import TwoGoalEnv
from bc_policy import BCPolicy, BCPolicyWithGaze
from gaze_simulator import GazeSimulator

def load_trained_models(model_dir='models'):
    """
    Load trained BC models with and without gaze information.
    
    Args:
        model_dir: Directory containing trained models
    
    Returns:
        bc_model: BC model without gaze
        bc_with_gaze_model: BC model with gaze
    """
    # Load model without gaze
    bc_model = BCPolicy(state_dim=2, hidden_dim=64, action_dim=2)
    bc_model_path = os.path.join(model_dir, 'bc_policy_best.pt')
    
    if os.path.exists(bc_model_path):
        bc_model.load_state_dict(torch.load(bc_model_path))
        print(f"Loaded BC model from {bc_model_path}")
    else:
        print(f"Warning: Could not find model at {bc_model_path}")
    
    # Load model with gaze
    bc_with_gaze_model = BCPolicyWithGaze(state_dim=2, gaze_dim=2, hidden_dim=64, action_dim=2)
    bc_with_gaze_model_path = os.path.join(model_dir, 'bc_policy_with_gaze_best.pt')
    
    if os.path.exists(bc_with_gaze_model_path):
        bc_with_gaze_model.load_state_dict(torch.load(bc_with_gaze_model_path))
        print(f"Loaded BC with gaze model from {bc_with_gaze_model_path}")
    else:
        print(f"Warning: Could not find model at {bc_with_gaze_model_path}")
    
    return bc_model, bc_with_gaze_model

def evaluate_model(model, with_gaze=False, num_episodes=100, render=False):
    """
    Evaluate a trained model on the environment.
    
    Args:
        model: Trained model (BCPolicy or BCPolicyWithGaze)
        with_gaze: Whether the model uses gaze information
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    env = TwoGoalEnv()
    
    if with_gaze:
        gaze_simulator = GazeSimulator(env, gaze_noise_level=0.2)
    
    # Metrics to track
    success_count = 0
    episode_lengths = []
    trajectory_lengths = []
    
    # Set model to evaluation mode
    model.eval()
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=episode)
        target_goal = info['target_goal']
        done = False
        steps = 0
        distance_traveled = 0
        prev_state = state.copy()
        
        if render and episode == 0:
            env.render()
        
        while not done:
            # Get action based on model type
            if with_gaze:
                # Generate gaze
                gaze = gaze_simulator.generate_gaze(state, target_goal)
                
                # Convert to tensors
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                gaze_tensor = torch.FloatTensor(gaze).unsqueeze(0)
                
                # Get action
                with torch.no_grad():
                    action = model(state_tensor, gaze_tensor).squeeze(0).numpy()
            else:
                # Convert to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action
                with torch.no_grad():
                    action = model(state_tensor).squeeze(0).numpy()
            
            # Take action in environment
            next_state, reward, done, _, info = env.step(action)
            
            if render and episode == 0:
                env.render()
            
            # Calculate distance traveled in this step
            step_distance = np.linalg.norm(next_state - state)
            distance_traveled += step_distance
            
            # Update state
            state = next_state
            steps += 1
        
        # Check if agent reached the correct goal
        if reward == 1.0:
            success_count += 1
        
        episode_lengths.append(steps)
        trajectory_lengths.append(distance_traveled)
    
    # Calculate metrics
    success_rate = success_count / num_episodes
    avg_episode_length = np.mean(episode_lengths)
    avg_trajectory_length = np.mean(trajectory_lengths)
    
    metrics = {
        'success_rate': success_rate,
        'avg_episode_length': avg_episode_length,
        'avg_trajectory_length': avg_trajectory_length,
        'episode_lengths': episode_lengths,
        'trajectory_lengths': trajectory_lengths
    }
    
    # Print results
    print(f"Model {'with' if with_gaze else 'without'} gaze:")
    print(f"  Success Rate: {success_rate:.4f}")
    print(f"  Average Episode Length: {avg_episode_length:.2f} steps")
    print(f"  Average Trajectory Length: {avg_trajectory_length:.4f} units")
    
    return metrics

def plot_comparison(bc_metrics, bc_with_gaze_metrics, save_path='results'):
    """
    Plot comparison of metrics between models.
    
    Args:
        bc_metrics: Metrics for BC model without gaze
        bc_with_gaze_metrics: Metrics for BC model with gaze
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Plot success rate comparison
    plt.figure(figsize=(12, 5))
    
    # Success rate bar chart
    plt.subplot(1, 3, 1)
    models = ['BC without Gaze', 'BC with Gaze']
    success_rates = [bc_metrics['success_rate'], bc_with_gaze_metrics['success_rate']]
    
    plt.bar(models, success_rates, color=['blue', 'green'])
    plt.title('Success Rate Comparison')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.0)
    
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 0.02, f"{rate:.3f}", ha='center')
    
    # Episode length comparison
    plt.subplot(1, 3, 2)
    avg_lengths = [bc_metrics['avg_episode_length'], bc_with_gaze_metrics['avg_episode_length']]
    
    plt.bar(models, avg_lengths, color=['blue', 'green'])
    plt.title('Average Episode Length')
    plt.ylabel('Steps')
    
    for i, length in enumerate(avg_lengths):
        plt.text(i, length + 0.5, f"{length:.1f}", ha='center')
    
    # Trajectory length comparison
    plt.subplot(1, 3, 3)
    avg_traj_lengths = [bc_metrics['avg_trajectory_length'], bc_with_gaze_metrics['avg_trajectory_length']]
    
    plt.bar(models, avg_traj_lengths, color=['blue', 'green'])
    plt.title('Average Trajectory Length')
    plt.ylabel('Distance')
    
    for i, length in enumerate(avg_traj_lengths):
        plt.text(i, length + 0.05, f"{length:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_comparison.png'), dpi=300)
    plt.close()
    
    # Plot episode length distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(bc_metrics['episode_lengths'], bins=15, alpha=0.7, label='BC without Gaze')
    plt.hist(bc_with_gaze_metrics['episode_lengths'], bins=15, alpha=0.7, label='BC with Gaze')
    plt.title('Episode Length Distribution')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(bc_metrics['trajectory_lengths'], bins=15, alpha=0.7, label='BC without Gaze')
    plt.hist(bc_with_gaze_metrics['trajectory_lengths'], bins=15, alpha=0.7, label='BC with Gaze')
    plt.title('Trajectory Length Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'length_distributions.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Load trained models
    bc_model, bc_with_gaze_model = load_trained_models()
    
    # Evaluate models
    print("Evaluating BC model without gaze...")
    bc_metrics = evaluate_model(bc_model, with_gaze=False, num_episodes=100)
    
    print("\nEvaluating BC model with gaze...")
    bc_with_gaze_metrics = evaluate_model(bc_with_gaze_model, with_gaze=True, num_episodes=100)
    
    # Plot comparison
    plot_comparison(bc_metrics, bc_with_gaze_metrics)
    
    # Print improvement
    improvement = (bc_with_gaze_metrics['success_rate'] - bc_metrics['success_rate']) / bc_metrics['success_rate'] * 100
    print(f"\nImprovement with gaze: {improvement:.2f}%") 