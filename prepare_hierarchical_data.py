import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class HierarchicalTrajectoryDataset(Dataset):
    """
    Dataset for training hierarchical behavioral cloning models.
    Provides (state, future_state, action_sequence) tuples.
    """
    
    def __init__(self, states, future_states, action_sequences):
        """
        Initialize dataset.
        
        Args:
            states: Current states [N, state_dim]
            future_states: Future states (subgoals) [N, state_dim]
            action_sequences: Sequences of actions to reach subgoals [N, seq_len, action_dim]
        """
        self.states = torch.FloatTensor(states)
        self.future_states = torch.FloatTensor(future_states)
        self.action_sequences = torch.FloatTensor(action_sequences)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'future_state': self.future_states[idx],
            'action_sequence': self.action_sequences[idx]
        }

def create_hierarchical_dataset(
    csv_path='data/trajectories.csv', 
    subgoal_horizon=10,
    sequence_length=5,
    save_path='data',
    test_split=0.2
):
    """
    Process trajectory data to create a dataset for hierarchical BC.
    
    Args:
        csv_path: Path to the trajectory data CSV
        subgoal_horizon: How many steps ahead to consider for subgoals
        sequence_length: Length of action sequences
        save_path: Directory to save processed data
        test_split: Fraction of data to use for test set
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Load the trajectory data
    df = pd.read_csv(csv_path)
    
    # Extract episode boundaries
    episode_ends = np.where(df['done'])[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    
    # Initialize lists to store dataset samples
    all_states = []
    all_future_states = []
    all_action_sequences = []
    
    # Process each episode
    for start, end in zip(episode_starts, episode_ends):
        # Get episode data
        episode_df = df.iloc[start:end+1]
        
        # Extract states and actions
        states = episode_df[['state_x', 'state_y']].values
        actions = episode_df[['action_x', 'action_y']].values
        
        # Create dataset samples with overlapping sequences
        for i in range(len(states) - subgoal_horizon):
            # Current state
            current_state = states[i]
            
            # Future state (subgoal) is subgoal_horizon steps ahead
            future_state = states[i + subgoal_horizon]
            
            # Get action sequence (up to sequence_length actions)
            seq_end = min(i + sequence_length, len(actions))
            action_seq = actions[i:seq_end]
            
            # If we have fewer than sequence_length actions, pad with zeros
            if len(action_seq) < sequence_length:
                padding = np.zeros((sequence_length - len(action_seq), 2))
                action_seq = np.vstack([action_seq, padding])
            
            # Add to lists
            all_states.append(current_state)
            all_future_states.append(future_state)
            all_action_sequences.append(action_seq)
    
    # Convert to arrays
    all_states = np.array(all_states)
    all_future_states = np.array(all_future_states)
    all_action_sequences = np.array(all_action_sequences)
    
    # Save processed data
    np.savez(
        os.path.join(save_path, 'hierarchical_data.npz'),
        states=all_states,
        future_states=all_future_states,
        action_sequences=all_action_sequences
    )
    
    # Create train/test split
    n_samples = len(all_states)
    n_test = int(n_samples * test_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Create datasets
    train_dataset = HierarchicalTrajectoryDataset(
        all_states[train_indices],
        all_future_states[train_indices],
        all_action_sequences[train_indices]
    )
    
    test_dataset = HierarchicalTrajectoryDataset(
        all_states[test_indices],
        all_future_states[test_indices],
        all_action_sequences[test_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Created dataset with {n_samples} samples")
    print(f"Training set: {len(train_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    
    return train_loader, test_loader


def create_hierarchical_dataset_with_gaze(
    csv_path='data/trajectories_with_gaze.csv', 
    subgoal_horizon=10,
    sequence_length=5,
    save_path='data',
    test_split=0.2
):
    """
    Process trajectory data with gaze to create a dataset for hierarchical BC.
    
    Args:
        csv_path: Path to the trajectory data CSV with gaze information
        subgoal_horizon: How many steps ahead to consider for subgoals
        sequence_length: Length of action sequences
        save_path: Directory to save processed data
        test_split: Fraction of data to use for test set
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # This is similar to the above function, but includes gaze data
    # for conditioning the subgoal prediction
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Load the trajectory data
    df = pd.read_csv(csv_path)
    
    # Extract episode boundaries
    episode_ends = np.where(df['done'])[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    
    # Initialize lists to store dataset samples
    all_states = []
    all_gazes = []
    all_future_states = []
    all_action_sequences = []
    
    # Process each episode
    for start, end in zip(episode_starts, episode_ends):
        # Get episode data
        episode_df = df.iloc[start:end+1]
        
        # Extract states, gazes, and actions
        states = episode_df[['state_x', 'state_y']].values
        gazes = episode_df[['gaze_x', 'gaze_y']].values
        actions = episode_df[['action_x', 'action_y']].values
        
        # Create dataset samples with overlapping sequences
        for i in range(len(states) - subgoal_horizon):
            # Current state and gaze
            current_state = states[i]
            current_gaze = gazes[i]
            
            # Future state (subgoal) is subgoal_horizon steps ahead
            future_state = states[i + subgoal_horizon]
            
            # Get action sequence (up to sequence_length actions)
            seq_end = min(i + sequence_length, len(actions))
            action_seq = actions[i:seq_end]
            
            # If we have fewer than sequence_length actions, pad with zeros
            if len(action_seq) < sequence_length:
                padding = np.zeros((sequence_length - len(action_seq), 2))
                action_seq = np.vstack([action_seq, padding])
            
            # Add to lists
            all_states.append(current_state)
            all_gazes.append(current_gaze)
            all_future_states.append(future_state)
            all_action_sequences.append(action_seq)
    
    # Convert to arrays
    all_states = np.array(all_states)
    all_gazes = np.array(all_gazes)
    all_future_states = np.array(all_future_states)
    all_action_sequences = np.array(all_action_sequences)
    
    # Save processed data
    np.savez(
        os.path.join(save_path, 'hierarchical_data_with_gaze.npz'),
        states=all_states,
        gazes=all_gazes,
        future_states=all_future_states,
        action_sequences=all_action_sequences
    )
    
    # Create train/test split
    n_samples = len(all_states)
    n_test = int(n_samples * test_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # For the hierarchical model with gaze, we'll let the main training
    # script handle the DataLoader creation, as we need to create
    # a custom Dataset class that handles the gaze information
    
    return {
        'train': {
            'states': all_states[train_indices],
            'gazes': all_gazes[train_indices],
            'future_states': all_future_states[train_indices],
            'action_sequences': all_action_sequences[train_indices]
        },
        'test': {
            'states': all_states[test_indices],
            'gazes': all_gazes[test_indices],
            'future_states': all_future_states[test_indices],
            'action_sequences': all_action_sequences[test_indices]
        }
    }


if __name__ == "__main__":
    # Create datasets
    print("Creating hierarchical dataset...")
    train_loader, test_loader = create_hierarchical_dataset()
    
    print("Creating hierarchical dataset with gaze...")
    data_with_gaze = create_hierarchical_dataset_with_gaze()
    
    print("Dataset preparation complete!") 