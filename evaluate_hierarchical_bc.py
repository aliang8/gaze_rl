import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from tqdm import tqdm
import argparse

from hierarchical_bc_models import HierarchicalBC
from environment import TwoGoalEnv
from gaze_simulator import GazeSimulator
from bayesian_gaze_model import BayesianGazeBC, GazeLikelihoodModel

def load_model(model_path, state_dim=2, action_dim=2, latent_dim=2, hidden_dim=64, sequence_length=5):
    """
    Load a trained hierarchical BC model.
    
    Args:
        model_path: Directory containing the saved model files
        state_dim: State dimension
        action_dim: Action dimension
        latent_dim: Latent dimension for CVAE
        hidden_dim: Hidden dimension for neural networks
        sequence_length: Sequence length for LSTM
    
    Returns:
        The loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = HierarchicalBC(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        device=device
    )
    
    # Load subgoal predictor if available
    subgoal_predictor_path = os.path.join(model_path, 'subgoal_predictor.pt')
    if os.path.exists(subgoal_predictor_path):
        model.subgoal_predictor.load_state_dict(torch.load(subgoal_predictor_path, map_location=device))
        print(f"Loaded subgoal predictor from {subgoal_predictor_path}")
    
    # Load policy
    policy_path = os.path.join(model_path, 'subgoal_policy.pt')
    if os.path.exists(policy_path):
        model.subgoal_policy.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"Loaded subgoal policy from {policy_path}")
    
    model.eval_mode()
    return model

def load_model_with_gaze(model_path, state_dim=2, action_dim=2, latent_dim=2, hidden_dim=64, sequence_length=5):
    """
    Load a trained hierarchical BC model with gaze.
    
    Args:
        model_path: Directory containing the saved model files
        state_dim: State dimension
        action_dim: Action dimension
        latent_dim: Latent dimension for CVAE
        hidden_dim: Hidden dimension for neural networks
        sequence_length: Sequence length for LSTM
    
    Returns:
        The loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = HierarchicalBC(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        device=device
    )
    
    # Load policy with gaze
    policy_path = os.path.join(model_path, 'subgoal_policy_with_gaze.pt')
    if os.path.exists(policy_path):
        model.subgoal_policy.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"Loaded subgoal policy with gaze from {policy_path}")
    else:
        print(f"Policy model with gaze not found at {policy_path}")
    
    model.eval_mode()
    return model

def load_bayesian_model(model_path, state_dim=2, action_dim=2, latent_dim=2, hidden_dim=64, sequence_length=5, num_samples=10, uncertainty_threshold=0.05):
    """
    Load a trained Bayesian gaze BC model.
    
    Args:
        model_path: Directory containing the saved model files
        state_dim: State dimension
        action_dim: Action dimension
        latent_dim: Latent dimension for CVAE
        hidden_dim: Hidden dimension for neural networks
        sequence_length: Sequence length for LSTM
        num_samples: Number of samples for Bayesian inference
        uncertainty_threshold: Threshold for when to use gaze based on uncertainty
    
    Returns:
        The loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create CVAE and subgoal policy models
    from hierarchical_bc_models import SubgoalPredictorCVAE, SubgoalReachingPolicy
    
    cvae = SubgoalPredictorCVAE(state_dim=state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    subgoal_policy = SubgoalReachingPolicy(state_dim=state_dim, 
                                          action_dim=action_dim, hidden_dim=hidden_dim, 
                                          sequence_length=sequence_length)
    
    # Create gaze likelihood model
    gaze_model = GazeLikelihoodModel(state_dim=state_dim, subgoal_dim=state_dim, hidden_dim=hidden_dim)
    
    # Load models
    cvae_path = os.path.join('models', 'subgoal_predictor.pt')
    policy_path = os.path.join('models', 'subgoal_policy.pt')
    gaze_model_path = os.path.join('results/hierarchical/bayesian', 'gaze_likelihood_model.pt')
    
    # Check if all required model files exist
    if not os.path.exists(cvae_path):
        raise FileNotFoundError(f"CVAE model not found at {cvae_path}")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Subgoal policy model not found at {policy_path}")
    if not os.path.exists(gaze_model_path):
        raise FileNotFoundError(f"Gaze likelihood model not found at {gaze_model_path}")
    
    # Create Bayesian gaze BC model
    bayesian_model = BayesianGazeBC(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        device=device,
        num_samples=num_samples,
        uncertainty_threshold=uncertainty_threshold
    )
    
    # Load model weights
    bayesian_model.subgoal_predictor.load_state_dict(torch.load(cvae_path, map_location=device))
    bayesian_model.subgoal_policy.load_state_dict(torch.load(policy_path, map_location=device))
    bayesian_model.gaze_likelihood_model.load_state_dict(torch.load(gaze_model_path, map_location=device))
    bayesian_model.to(device)
    
    return bayesian_model

def generate_rollout(env, model, gaze_model=None, max_steps=100, use_gaze=False, use_bayesian=False):
    """
    Generate a rollout using the hierarchical BC model.
    
    Args:
        env: Environment to run the rollout in
        model: Hierarchical BC model
        gaze_model: Gaze simulator (optional)
        max_steps: Maximum number of steps
        use_gaze: Whether to use gaze as subgoal
        use_bayesian: Whether to use Bayesian inference with gaze
    
    Returns:
        Dictionary containing trajectory data
    """
    # Access device directly from model
    device = model.device
    
    # Reset environment
    state, info = env.reset()
    target_goal = info['target_goal']  # Get the target goal from the environment
    done = False
    step = 0
    
    # Initialize trajectory data
    states = [state]
    actions = []
    subgoals = []
    gazes = []
    gaze_used_flags = []  # Track when gaze was actually used
    uncertainties = []    # Track subgoal prediction uncertainty
    
    # For Bayesian visualization
    all_sampled_subgoals = []  # Store all sampled subgoals at each step
    selected_indices = []      # Store which subgoal was selected
    
    # Get the uncertainty threshold if using Bayesian model
    uncertainty_threshold = getattr(model, 'uncertainty_threshold', 0.05) if use_bayesian else 0.05
    
    # Reset the hierarchical policy's internal state
    model.current_subgoal = None
    model.steps_to_next_subgoal = 0
    model.lstm_hidden = None
    
    while not done and step < max_steps:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Default values
        gaze = None
        used_gaze = False
        uncertainty = 0.0
        
        if use_gaze and gaze_model is not None:
            # Generate gaze
            gaze = gaze_model.generate_gaze(state, target_goal)
            gazes.append(gaze)
            
            if use_bayesian and isinstance(model, BayesianGazeBC):
                # Use Bayesian inference with uncertainty-based gaze
                gaze_tensor = torch.FloatTensor(gaze).unsqueeze(0).to(device)
                with torch.no_grad():
                    # Call the BayesianGazeBC's new predict_action method that returns (action, used_gaze)
                    action, used_gaze = model.predict_action(state_tensor, gaze_tensor)
                    
                    # Convert action to numpy
                    if torch.is_tensor(action):
                        action = action.cpu().numpy()
                    
                    # Store the uncertainty if available
                    if hasattr(model, 'last_uncertainty'):
                        uncertainty = model.last_uncertainty
                    
                    # Store sampled subgoals and selected index for visualization
                    if hasattr(model, 'last_sampled_subgoals') and model.last_sampled_subgoals is not None:
                        all_sampled_subgoals.append(model.last_sampled_subgoals[0])  # First batch item
                        selected_indices.append(model.last_selected_idx[0] if model.last_selected_idx is not None else None)
                    
                    # Ensure action has shape (2,) not (1, 2)
                    if isinstance(action, np.ndarray) and len(action.shape) > 1 and action.shape[0] == 1:
                        action = action.squeeze(0)
            else:
                # Use the HierarchicalBC's predict_action method with gaze
                gaze_tensor = torch.FloatTensor(gaze).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model.predict_action(state_tensor, gaze_tensor, reset=(step==0))
                    used_gaze = True
                
                # Ensure action has shape (2,) not (1, 2)
                if isinstance(action, np.ndarray) and len(action.shape) > 1 and action.shape[0] == 1:
                    action = action.squeeze(0)
        else:
            # Use the HierarchicalBC's predict_action method without gaze
            with torch.no_grad():
                action = model.predict_action(state_tensor, reset=(step==0))
                # Record the subgoal for visualization
                if model.current_subgoal is not None:
                    subgoals.append(model.current_subgoal.cpu().numpy()[0])
            
            # Ensure action has shape (2,) not (1, 2)
            if isinstance(action, np.ndarray) and len(action.shape) > 1 and action.shape[0] == 1:
                action = action.squeeze(0)
        
        # Take step in environment
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated  # Combine terminated and truncated for done state
        
        # Store data
        actions.append(action)
        states.append(next_state)
        gaze_used_flags.append(used_gaze)
        uncertainties.append(uncertainty)
        
        # Update state
        state = next_state
        step += 1
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'subgoals': np.array(subgoals) if subgoals else None,
        'gazes': np.array(gazes) if gazes else None,
        'gaze_used': np.array(gaze_used_flags),
        'uncertainties': np.array(uncertainties),
        'uncertainty_threshold': uncertainty_threshold,  # Store the threshold used
        'all_sampled_subgoals': all_sampled_subgoals if all_sampled_subgoals else None,  # All sampled subgoals
        'selected_indices': selected_indices if selected_indices else None,  # Which one was selected
        'success': reward > 0,  # Success if reward is positive (reached the correct goal)
        'steps': step
    }

def evaluate_model(env, model, gaze_model=None, n_episodes=50, max_steps=100, use_gaze=False, use_bayesian=False):
    """
    Evaluate a hierarchical BC model on multiple episodes.
    
    Args:
        env: Environment to evaluate in
        model: Hierarchical BC model
        gaze_model: Gaze simulator (optional)
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        use_gaze: Whether to use gaze as subgoal
        use_bayesian: Whether to use Bayesian inference with gaze
    
    Returns:
        Dictionary of evaluation metrics
    """
    success_rate = 0
    episode_lengths = []
    trajectories = []
    
    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        rollout = generate_rollout(env, model, gaze_model, max_steps, use_gaze, use_bayesian)
        success_rate += rollout['success']
        episode_lengths.append(rollout['steps'])
        trajectories.append(rollout)
    
    success_rate /= n_episodes
    avg_episode_length = np.mean(episode_lengths)
    
    return {
        'success_rate': success_rate,
        'avg_episode_length': avg_episode_length,
        'trajectories': trajectories
    }

def visualize_trajectory(trajectory, env, title="Agent Trajectory", save_path=None, show_subgoals=True, show_gaze=True):
    """
    Visualize a trajectory as a static plot.
    
    Args:
        trajectory: Trajectory data
        env: Environment
        title: Plot title
        save_path: Path to save the plot
        show_subgoals: Whether to show subgoals
        show_gaze: Whether to show gaze points
    """
    states = trajectory['states']
    subgoals = trajectory['subgoals'] if 'subgoals' in trajectory and trajectory['subgoals'] is not None else None
    gazes = trajectory['gazes'] if 'gazes' in trajectory and trajectory['gazes'] is not None else None
    gaze_used = trajectory.get('gaze_used', None)
    uncertainties = trajectory.get('uncertainties', None)
    uncertainty_threshold = trajectory.get('uncertainty_threshold', 0.05)
    
    # Bayesian visualization data
    all_sampled_subgoals = trajectory.get('all_sampled_subgoals', None)
    selected_indices = trajectory.get('selected_indices', None)
    
    # Create a figure with two subplots if we have uncertainty data
    if uncertainties is not None and len(uncertainties) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Plot environment
    env.render_on_axes(ax1)
    
    # Plot agent trajectory
    ax1.plot(states[:, 0], states[:, 1], 'b-', label='Agent Path', linewidth=2)
    ax1.scatter(states[0, 0], states[0, 1], c='g', s=100, label='Start')
    ax1.scatter(states[-1, 0], states[-1, 1], c='r', s=100, label='End')
    
    # Plot all sampled subgoals for Bayesian model if available
    if all_sampled_subgoals is not None and len(all_sampled_subgoals) > 0:
        # Choose a subset of steps to visualize (to avoid overcrowding)
        visualization_steps = min(5, len(all_sampled_subgoals))
        step_indices = np.linspace(0, len(all_sampled_subgoals)-1, visualization_steps, dtype=int)
        
        # Create a custom colormap for visualizing different time steps
        cmap = plt.cm.get_cmap('viridis', visualization_steps)
        
        for i, step_idx in enumerate(step_indices):
            step_subgoals = all_sampled_subgoals[step_idx]
            selected_idx = selected_indices[step_idx] if selected_indices[step_idx] is not None else None
            
            # Plot all sampled subgoals for this step
            for j in range(len(step_subgoals)):
                # Decide marker and size based on whether this was the selected subgoal
                is_selected = selected_idx is not None and j == selected_idx
                marker_size = 80 if is_selected else 20
                marker_alpha = 0.8 if is_selected else 0.3
                marker_style = 'o' if is_selected else 'x'
                
                # Use the same color for all subgoals from the same step, but different marker for the selected one
                subgoal_color = cmap(i)
                
                # Plot the subgoal
                ax1.scatter(step_subgoals[j, 0], step_subgoals[j, 1], 
                          c=[subgoal_color], s=marker_size, alpha=marker_alpha, 
                          marker=marker_style)
                
                # For the first step, add labels to the legend
                if i == 0 and j == 0:
                    ax1.scatter([], [], c='gray', s=20, alpha=0.3, marker='x', 
                              label='Sampled Subgoals')
                    ax1.scatter([], [], c='gray', s=80, alpha=0.8, marker='o', 
                              label='Selected Subgoal')
    
    # Plot regular subgoals if available and not showing Bayesian subgoals
    elif show_subgoals and subgoals is not None and len(subgoals) > 0:
        ax1.scatter(subgoals[:, 0], subgoals[:, 1], c='purple', s=50, alpha=0.5, label='Predicted Subgoals')
    
    # Plot gaze if available
    if show_gaze and gazes is not None and len(gazes) > 0:
        # If we have gaze_used flags, color gaze points differently based on whether they were used
        if gaze_used is not None and len(gaze_used) == len(gazes):
            # Create empty arrays for used and unused gazes
            used_gazes = []
            unused_gazes = []
            
            for i, (gaze, used) in enumerate(zip(gazes, gaze_used)):
                if used:
                    used_gazes.append(gaze)
                else:
                    unused_gazes.append(gaze)
            
            if used_gazes:
                ax1.scatter(np.array(used_gazes)[:, 0], np.array(used_gazes)[:, 1], 
                           c='orange', s=70, alpha=0.7, label='Used Gaze Points')
            
            if unused_gazes:
                ax1.scatter(np.array(unused_gazes)[:, 0], np.array(unused_gazes)[:, 1], 
                           c='gray', s=30, alpha=0.4, label='Unused Gaze Points')
        else:
            # Default behavior if we don't have gaze_used flags
            ax1.scatter(gazes[:, 0], gazes[:, 1], c='orange', s=50, alpha=0.5, label='Gaze Points')
    
    ax1.legend()
    ax1.set_title(title)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    
    # Plot uncertainty over time if available
    if uncertainties is not None and len(uncertainties) > 0:
        time_steps = np.arange(len(uncertainties))
        ax2.plot(time_steps, uncertainties, 'r-', linewidth=2)
        
        # Plot the threshold line
        ax2.axhline(y=uncertainty_threshold, color='k', linestyle='--', 
                  label=f'Threshold ({uncertainty_threshold:.3f})')
        
        # Add markers for when gaze was used
        if gaze_used is not None:
            for i, used in enumerate(gaze_used):
                if used:
                    ax2.scatter([i], [uncertainties[i]], color='orange', s=50, zorder=5)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Subgoal Uncertainty')
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved trajectory plot to {save_path}")
    
    plt.close()

def create_animation(trajectory, env, title="Agent Trajectory", save_path=None, show_subgoals=True, show_gaze=True):
    """
    Create an animation of a trajectory.
    
    Args:
        trajectory: Trajectory data
        env: Environment
        title: Animation title
        save_path: Path to save the animation
        show_subgoals: Whether to show subgoals
        show_gaze: Whether to show gaze points
    """
    states = trajectory['states']
    subgoals = trajectory['subgoals'] if 'subgoals' in trajectory and trajectory['subgoals'] is not None else None
    gazes = trajectory['gazes'] if 'gazes' in trajectory and trajectory['gazes'] is not None else None
    gaze_used = trajectory.get('gaze_used', None)
    
    # Bayesian visualization data
    all_sampled_subgoals = trajectory.get('all_sampled_subgoals', None)
    selected_indices = trajectory.get('selected_indices', None)
    
    # Parameters for the animation
    pause_frames = 3  # Number of frames to pause when showing sampled subgoals
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initial setup
    env.render_on_axes(ax)
    
    # Initialize plots
    line, = ax.plot([], [], 'b-', linewidth=2, label='Agent Path')
    agent_point, = ax.plot([], [], 'bo', markersize=10)
    start_point = ax.scatter(states[0, 0], states[0, 1], c='g', s=100, label='Start')
    
    # Setup for subgoals and gaze
    subgoal_points = None
    if show_subgoals and subgoals is not None and len(subgoals) > 0:
        subgoal_points, = ax.plot([], [], 'purple', marker='o', linestyle='', markersize=8, alpha=0.5, label='Predicted Subgoals')
    
    gaze_point = None
    if show_gaze and gazes is not None and len(gazes) > 0:
        if gaze_used is not None:
            # Create separate scatter for used and unused gaze
            used_gaze_points = ax.scatter([], [], c='orange', s=70, alpha=0.7, marker='o', label='Used Gaze Point')
            unused_gaze_points = ax.scatter([], [], c='gray', s=30, alpha=0.4, marker='o', label='Unused Gaze Point')
            gaze_point = (used_gaze_points, unused_gaze_points)
        else:
            gaze_point = ax.scatter([], [], c='orange', s=50, alpha=0.5, marker='o', label='Gaze Point')
    
    # Setup for Bayesian sampled subgoals
    sampled_subgoals_scatter = None
    selected_subgoal_scatter = None
    
    if all_sampled_subgoals is not None and len(all_sampled_subgoals) > 0:
        # Initialize empty scatter plots for sampled and selected subgoals
        sampled_subgoals_scatter = ax.scatter([], [], c='gray', s=20, alpha=0.3, marker='x', label='Sampled Subgoals')
        selected_subgoal_scatter = ax.scatter([], [], c='gray', s=80, alpha=0.8, marker='o', label='Selected Subgoal')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.legend()
    
    # Store key frames where we want to pause (frames with sampled subgoals)
    key_frames = set()
    if all_sampled_subgoals is not None:
        # Choose a subset of steps to visualize sampled subgoals
        visualization_steps = min(5, len(all_sampled_subgoals))
        step_indices = np.linspace(0, len(all_sampled_subgoals)-1, visualization_steps, dtype=int)
        key_frames = set(step_indices)
    
    # Create a colormap for visualizing different time steps
    cmap = plt.cm.get_cmap('viridis', max(5, len(states)))
    
    # Animation update function
    def update(frame_idx):
        # Determine actual frame to display (accounting for pauses)
        frame = min(frame_idx // (pause_frames if frame_idx // pause_frames in key_frames else 1), len(states) - 1)
        
        # Update agent trajectory
        line.set_data(states[:frame+1, 0], states[:frame+1, 1])
        agent_point.set_data(states[frame, 0], states[frame, 1])
        
        # Update subgoal if available
        artists = [line, agent_point]
        if show_subgoals and subgoal_points is not None and frame < len(subgoals):
            subgoal_points.set_data([subgoals[frame, 0]], [subgoals[frame, 1]])
            artists.append(subgoal_points)
        
        # Update gaze if available
        if show_gaze and gaze_point is not None and frame < len(gazes):
            if isinstance(gaze_point, tuple):  # We have separate used/unused gaze
                # Clear previous gaze points
                used_gaze_points, unused_gaze_points = gaze_point
                used_gaze_points.set_offsets(np.empty((0, 2)))
                unused_gaze_points.set_offsets(np.empty((0, 2)))
                
                # Set current gaze point based on whether it was used
                if gaze_used[frame]:
                    used_gaze_points.set_offsets([gazes[frame]])
                else:
                    unused_gaze_points.set_offsets([gazes[frame]])
                
                artists.extend([used_gaze_points, unused_gaze_points])
            else:
                gaze_point.set_offsets([gazes[frame]])
                artists.append(gaze_point)
        
        # Update Bayesian sampled subgoals if available
        if all_sampled_subgoals is not None and frame in key_frames and sampled_subgoals_scatter is not None:
            step_idx = list(key_frames).index(frame)
            if step_idx < len(all_sampled_subgoals):
                subgoals_data = all_sampled_subgoals[frame]
                
                # Update sampled subgoals
                x_coords = []
                y_coords = []
                colors = []
                
                # Clear previous data
                sampled_subgoals_scatter.set_offsets(np.empty((0, 2)))
                selected_subgoal_scatter.set_offsets(np.empty((0, 2)))
                
                # Add all sampled subgoals
                for i in range(len(subgoals_data)):
                    x_coords.append(subgoals_data[i, 0])
                    y_coords.append(subgoals_data[i, 1])
                    colors.append(cmap(frame / len(states)))
                
                # Update the scatter plot
                sampled_subgoals_scatter.set_offsets(np.column_stack([x_coords, y_coords]))
                sampled_subgoals_scatter.set_color(colors)
                
                # Update selected subgoal if applicable
                if selected_indices is not None and frame < len(selected_indices) and selected_indices[frame] is not None:
                    selected_idx = selected_indices[frame]
                    selected_subgoal_scatter.set_offsets([[subgoals_data[selected_idx, 0], subgoals_data[selected_idx, 1]]])
                    selected_subgoal_scatter.set_color(cmap(frame / len(states)))
                
                artists.extend([sampled_subgoals_scatter, selected_subgoal_scatter])
        
        return tuple(artists)
    
    # Calculate total number of frames, accounting for pauses at key frames
    total_frames = len(states) - 1
    if key_frames:
        # Add pause_frames-1 extra frames for each key frame
        total_frames += (pause_frames - 1) * len(key_frames)
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=total_frames, blit=True, interval=200)
    
    # Save animation
    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=5)
        print(f"Saved animation to {save_path}")
    
    plt.close()

def compare_models(standard_results, gaze_results, save_dir='results', 
                  labels=['Standard Hierarchical BC', 'Hierarchical BC with Gaze']):
    """
    Compare the performance of two BC models.
    
    Args:
        standard_results: Results from the first model
        gaze_results: Results from the second model
        save_dir: Directory to save comparison results
        labels: List of two strings with model names for plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compare success rates
    success_rates = [standard_results['success_rate'], gaze_results['success_rate']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, success_rates, color=['blue', 'orange'])
    plt.ylim(0, 1.05)
    plt.title('Success Rate Comparison')
    plt.ylabel('Success Rate')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'success_rate_comparison.png'))
    plt.close()
    
    # Compare episode lengths
    episode_lengths = [standard_results['avg_episode_length'], gaze_results['avg_episode_length']]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, episode_lengths, color=['blue', 'orange'])
    plt.title('Average Episode Length Comparison')
    plt.ylabel('Steps')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'episode_length_comparison.png'))
    plt.close()
    
    # Save numeric results
    with open(os.path.join(save_dir, 'comparison_results.txt'), 'w') as f:
        f.write("Model Comparison\n")
        f.write("================================\n\n")
        f.write(f"{labels[0]}:\n")
        f.write(f"  Success Rate: {standard_results['success_rate']:.4f}\n")
        f.write(f"  Average Episode Length: {standard_results['avg_episode_length']:.2f}\n\n")
        f.write(f"{labels[1]}:\n")
        f.write(f"  Success Rate: {gaze_results['success_rate']:.4f}\n")
        f.write(f"  Average Episode Length: {gaze_results['avg_episode_length']:.2f}\n\n")
        f.write(f"Improvement with {labels[1]} over {labels[0]}:\n")
        sr_improvement = gaze_results['success_rate'] - standard_results['success_rate']
        sr_percent = (sr_improvement / max(standard_results['success_rate'], 1e-5)) * 100
        f.write(f"  Success Rate: {sr_improvement:.4f} ({sr_percent:.2f}%)\n")
        
        el_reduction = standard_results['avg_episode_length'] - gaze_results['avg_episode_length']
        el_percent = (el_reduction / max(standard_results['avg_episode_length'], 1e-5)) * 100
        f.write(f"  Episode Length Reduction: {el_reduction:.2f} ({el_percent:.2f}%)\n")

def evaluate_bayesian_thresholds(env, model_path, gaze_model, n_episodes=10, max_steps=100, 
                               thresholds=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5], save_dir='results/hierarchical/thresholds'):
    """
    Evaluate the Bayesian Gaze BC model with different uncertainty thresholds.
    
    Args:
        env: Environment to evaluate in
        model_path: Path to model directory
        gaze_model: Gaze simulator
        n_episodes: Number of episodes to evaluate per threshold
        max_steps: Maximum steps per episode
        thresholds: List of uncertainty thresholds to evaluate
        save_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics for each threshold
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    avg_gaze_usage = []
    success_rates = []
    episode_lengths = []
    
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        # Load model with this threshold
        model = load_bayesian_model(
            model_path,
            num_samples=100,
            uncertainty_threshold=threshold
        )
        
        # Evaluate model
        threshold_results = evaluate_model(
            env, model, gaze_model, n_episodes, max_steps, 
            use_gaze=True, use_bayesian=True
        )
        
        # Calculate average gaze usage
        gaze_usage = []
        for traj in threshold_results['trajectories']:
            if 'gaze_used' in traj:
                gaze_usage.append(np.mean(traj['gaze_used']))
        
        mean_gaze_usage = np.mean(gaze_usage) if gaze_usage else 0.0
        
        # Store results
        results[threshold] = {
            'success_rate': threshold_results['success_rate'],
            'avg_episode_length': threshold_results['avg_episode_length'],
            'avg_gaze_usage': mean_gaze_usage
        }
        
        # Store for plotting
        avg_gaze_usage.append(mean_gaze_usage)
        success_rates.append(threshold_results['success_rate'])
        episode_lengths.append(threshold_results['avg_episode_length'])
    
    # Plot results
    plot_threshold_comparison(thresholds, success_rates, episode_lengths, avg_gaze_usage, save_dir)
    
    return results

def plot_threshold_comparison(thresholds, success_rates, episode_lengths, avg_gaze_usage, save_dir):
    """
    Plot the comparison of different uncertainty thresholds.
    
    Args:
        thresholds: List of uncertainty thresholds
        success_rates: List of success rates for each threshold
        episode_lengths: List of average episode lengths for each threshold
        avg_gaze_usage: List of average gaze usage for each threshold
        save_dir: Directory to save the plots
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot success rate
    ax1.plot(thresholds, success_rates, 'bo-', linewidth=2)
    ax1.set_xlabel('Uncertainty Threshold')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate vs. Uncertainty Threshold')
    ax1.grid(True)
    
    # Plot episode length
    ax2.plot(thresholds, episode_lengths, 'ro-', linewidth=2)
    ax2.set_xlabel('Uncertainty Threshold')
    ax2.set_ylabel('Avg. Episode Length')
    ax2.set_title('Average Episode Length vs. Uncertainty Threshold')
    ax2.grid(True)
    
    # Plot gaze usage
    ax3.plot(thresholds, avg_gaze_usage, 'go-', linewidth=2)
    ax3.set_xlabel('Uncertainty Threshold')
    ax3.set_ylabel('Avg. Gaze Usage')
    ax3.set_title('Average Gaze Usage vs. Uncertainty Threshold')
    ax3.grid(True)
    ax3.set_ylim(0, 1.05)  # Gaze usage is a proportion between 0 and 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'threshold_comparison.png'))
    plt.close()
    
    # Save the data as CSV
    import pandas as pd
    df = pd.DataFrame({
        'threshold': thresholds,
        'success_rate': success_rates,
        'avg_episode_length': episode_lengths,
        'avg_gaze_usage': avg_gaze_usage
    })
    df.to_csv(os.path.join(save_dir, 'threshold_comparison.csv'), index=False)
    
    # Save a text summary
    with open(os.path.join(save_dir, 'threshold_comparison.txt'), 'w') as f:
        f.write("Uncertainty Threshold Comparison\n")
        f.write("===============================\n\n")
        
        for i, threshold in enumerate(thresholds):
            f.write(f"Threshold: {threshold}\n")
            f.write(f"  Success Rate: {success_rates[i]:.4f}\n")
            f.write(f"  Avg. Episode Length: {episode_lengths[i]:.2f}\n")
            f.write(f"  Avg. Gaze Usage: {avg_gaze_usage[i]:.4f} ({avg_gaze_usage[i]*100:.1f}%)\n\n")

def main(args):
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environment
    env = TwoGoalEnv()
    gaze_model = GazeSimulator(env)
    
    # If evaluating multiple thresholds
    if args.evaluate_thresholds:
        print("Evaluating Bayesian model with multiple uncertainty thresholds...")
        thresholds = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5] if args.thresholds is None else args.thresholds
        threshold_results = evaluate_bayesian_thresholds(
            env, args.model_dir, gaze_model, args.n_episodes, args.max_steps, 
            thresholds=thresholds, save_dir=os.path.join(args.save_dir, 'thresholds')
        )
        print(f"Threshold evaluation complete. Results saved to {os.path.join(args.save_dir, 'thresholds')}")
        return
    
    # Load models
    standard_model = load_model(args.model_dir)
    gaze_model_with_gaze = load_model_with_gaze(args.model_dir)
    
    # Load Bayesian model if requested
    bayesian_model = None
    if args.evaluate_bayesian:
        try:
            bayesian_model = load_bayesian_model(
                args.model_dir,
                num_samples=args.bayesian_samples,
                uncertainty_threshold=args.uncertainty_threshold
            )
            print(f"Loaded Bayesian gaze BC model with uncertainty threshold: {args.uncertainty_threshold}")
        except Exception as e:
            print(f"Failed to load Bayesian gaze BC model: {e}")
            print("Skipping Bayesian model evaluation")
    
    print("Evaluating standard hierarchical BC model...")
    standard_results = evaluate_model(env, standard_model, gaze_model, args.n_episodes, args.max_steps, use_gaze=False)
    
    print("Evaluating hierarchical BC model with gaze...")
    gaze_results = evaluate_model(env, gaze_model_with_gaze, gaze_model, args.n_episodes, args.max_steps, use_gaze=True)
    
    # Evaluate Bayesian model if available
    bayesian_results = None
    if bayesian_model is not None:
        print(f"Evaluating Bayesian gaze BC model (uncertainty threshold: {args.uncertainty_threshold})...")
        bayesian_results = evaluate_model(
            env, bayesian_model, gaze_model, args.n_episodes, args.max_steps, 
            use_gaze=True, use_bayesian=True
        )
    
    # Compare models
    compare_models(standard_results, gaze_results, args.save_dir)
    
    # Compare with Bayesian model if available
    if bayesian_results is not None:
        save_dir_bayesian = os.path.join(args.save_dir, 'bayesian')
        os.makedirs(save_dir_bayesian, exist_ok=True)
        
        # Compare standard vs Bayesian
        compare_models(standard_results, bayesian_results, save_dir_bayesian, 
                       labels=['Standard Hierarchical BC', 'Bayesian Gaze BC'])
        
        # Compare direct gaze vs Bayesian
        compare_models(gaze_results, bayesian_results, os.path.join(save_dir_bayesian, 'gaze_vs_bayesian'), 
                       labels=['Direct Gaze BC', 'Bayesian Gaze BC'])
    
    # Visualize a few trajectories
    for i in range(min(args.n_viz, args.n_episodes)):
        # Visualize standard model trajectory
        visualize_trajectory(
            standard_results['trajectories'][i],
            env,
            title=f"Standard Hierarchical BC Trajectory {i+1}",
            save_path=os.path.join(args.save_dir, f"standard_trajectory_{i+1}.png"),
            show_subgoals=True,
            show_gaze=True
        )
        
        # Visualize model with gaze trajectory
        visualize_trajectory(
            gaze_results['trajectories'][i],
            env,
            title=f"Hierarchical BC with Gaze Trajectory {i+1}",
            save_path=os.path.join(args.save_dir, f"gaze_trajectory_{i+1}.png"),
            show_subgoals=False,
            show_gaze=True
        )
        
        # Visualize Bayesian model trajectory if available
        if bayesian_results is not None:
            visualize_trajectory(
                bayesian_results['trajectories'][i],
                env,
                title=f"Bayesian Gaze BC (Threshold={args.uncertainty_threshold}) Trajectory {i+1}",
                save_path=os.path.join(args.save_dir, f"bayesian_trajectory_{i+1}.png"),
                show_subgoals=True,
                show_gaze=True
            )
        
        # Create animations
        if args.create_animations:
            create_animation(
                standard_results['trajectories'][i],
                env,
                title=f"Standard Hierarchical BC Trajectory {i+1}",
                save_path=os.path.join(args.save_dir, f"standard_animation_{i+1}.mp4"),
                show_subgoals=True,
                show_gaze=True
            )
            
            create_animation(
                gaze_results['trajectories'][i],
                env,
                title=f"Hierarchical BC with Gaze Trajectory {i+1}",
                save_path=os.path.join(args.save_dir, f"gaze_animation_{i+1}.mp4"),
                show_subgoals=False,
                show_gaze=True
            )
            
            # Create animation for Bayesian model if available
            if bayesian_results is not None:
                create_animation(
                    bayesian_results['trajectories'][i],
                    env,
                    title=f"Bayesian Gaze BC Trajectory {i+1}",
                    save_path=os.path.join(args.save_dir, f"bayesian_animation_{i+1}.mp4"),
                    show_subgoals=True,
                    show_gaze=True
                )
    
    print(f"Evaluation complete! Results saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hierarchical BC models")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--save-dir", type=str, default="results/hierarchical", help="Directory to save results")
    parser.add_argument("--n-episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--n-viz", type=int, default=5, help="Number of trajectories to visualize")
    parser.add_argument("--create-animations", action="store_true", help="Create trajectory animations")
    parser.add_argument("--evaluate-bayesian", action="store_true", help="Evaluate Bayesian gaze BC model")
    parser.add_argument("--bayesian-samples", type=int, default=10, help="Number of samples for Bayesian inference")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.05, help="Threshold for uncertainty-based gaze")
    parser.add_argument("--evaluate-thresholds", action="store_true", help="Evaluate multiple uncertainty thresholds")
    parser.add_argument("--thresholds", type=float, nargs="+", help="List of uncertainty thresholds to evaluate")
    
    args = parser.parse_args()
    main(args) 