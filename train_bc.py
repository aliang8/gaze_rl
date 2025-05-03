import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from bc_policy import BCPolicy
from environment import TwoGoalEnv

class TrajectoryDataset(Dataset):
    """Dataset for trajectory data."""
    
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def load_data(data_path='data/trajectories.csv'):
    """Load trajectory data from CSV file."""
    df = pd.read_csv(data_path)
    
    # Extract states and actions
    states = df[['state_x', 'state_y']].values
    actions = df[['action_x', 'action_y']].values
    
    return states, actions

def train_bc_policy(
    data_path='data/trajectories.csv',
    model_save_path='models',
    batch_size=64,
    learning_rate=1e-3,
    epochs=100,
    test_size=0.2,
    hidden_dim=64,
    seed=42
):
    """
    Train a Behavioral Cloning policy on the trajectory data.
    
    Args:
        data_path: Path to the trajectory data CSV file
        model_save_path: Directory to save model checkpoints
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        test_size: Fraction of data to use for validation
        hidden_dim: Hidden dimension of the policy network
        seed: Random seed for reproducibility
    
    Returns:
        model: Trained BC policy model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load data
    states, actions = load_data(data_path)
    
    # Split data into train and validation sets
    states_train, states_val, actions_train, actions_val = train_test_split(
        states, actions, test_size=test_size, random_state=seed
    )
    
    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(states_train, actions_train)
    val_dataset = TrajectoryDataset(states_val, actions_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    state_dim = states.shape[1]  # 2 (x, y)
    action_dim = actions.shape[1]  # 2 (dx, dy)
    
    model = BCPolicy(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for states_batch, actions_batch in train_loader:
            # Forward pass
            predicted_actions = model(states_batch)
            
            # Compute loss
            loss = criterion(predicted_actions, actions_batch)
            train_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for states_batch, actions_batch in val_loader:
                # Forward pass
                predicted_actions = model(states_batch)
                
                # Compute loss
                loss = criterion(predicted_actions, actions_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'bc_policy_best.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_save_path, 'bc_policy_final.pt'))
    
    # Plot training curve
    plot_training_curve(train_losses, val_losses, model_save_path)
    
    # Evaluate on test environment
    evaluate_policy(model, num_episodes=50, render=False)
    
    return model, train_losses, val_losses

def plot_training_curve(train_losses, val_losses, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'bc_training_curve.png'), dpi=300)
    plt.close()

def evaluate_policy(model, num_episodes=50, render=False):
    """
    Evaluate the trained policy on the test environment.
    
    Args:
        model: Trained BC policy model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    
    Returns:
        success_rate: Fraction of episodes where agent reached the correct goal
    """
    env = TwoGoalEnv()
    success_count = 0
    
    # Set model to evaluation mode
    model.eval()
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=episode)
        target_goal = info['target_goal']
        done = False
        
        if render and episode == 0:
            env.render()
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action = model(state_tensor).squeeze(0).numpy()
            
            # Take action in environment
            next_state, reward, done, _, info = env.step(action)
            
            if render and episode == 0:
                env.render()
            
            # Update state
            state = next_state
        
        # Check if agent reached the correct goal
        if reward == 1.0:
            success_count += 1
    
    success_rate = success_count / num_episodes
    print(f"Evaluation: Success Rate = {success_rate:.4f}")
    
    return success_rate

if __name__ == "__main__":
    # Train BC policy
    model, train_losses, val_losses = train_bc_policy(
        data_path='data/trajectories.csv',
        model_save_path='models',
        batch_size=64,
        learning_rate=1e-3,
        epochs=100,
        test_size=0.2,
        hidden_dim=64,
        seed=42
    ) 