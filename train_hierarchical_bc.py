import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_bc_models import HierarchicalBC
from prepare_hierarchical_data import create_hierarchical_dataset

def train_hierarchical_bc(
    data_path='data/hierarchical_data.npz', 
    model_save_path='models',
    batch_size=64,
    subgoal_epochs=100,
    policy_epochs=100,
    learning_rate=1e-3,
    latent_dim=2,
    hidden_dim=64,
    sequence_length=5,
    seed=42
):
    """
    Train a hierarchical BC model on trajectory data.
    
    Args:
        data_path: Path to the processed hierarchical data
        model_save_path: Directory to save the trained models
        batch_size: Batch size for training
        subgoal_epochs: Number of epochs to train the subgoal predictor
        policy_epochs: Number of epochs to train the low-level policy
        learning_rate: Learning rate for optimizers
        latent_dim: Dimension of the latent space in the CVAE
        hidden_dim: Hidden layer dimension
        sequence_length: Length of action sequences
        seed: Random seed
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model save directory
    os.makedirs(model_save_path, exist_ok=True)
    
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or create the dataset
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
        states = data['states']
        future_states = data['future_states']
        action_sequences = data['action_sequences']
    else:
        print(f"Data file not found. Creating new dataset...")
        train_loader, test_loader = create_hierarchical_dataset(save_path=os.path.dirname(data_path))
        # We'll recreate the loaders later, just need the data
        data = np.load(data_path)
        states = data['states']
        future_states = data['future_states']
        action_sequences = data['action_sequences']
    
    # Create train/test split
    n_samples = len(states)
    n_test = int(n_samples * 0.2)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    train_states = states[train_indices]
    train_future_states = future_states[train_indices]
    train_action_sequences = action_sequences[train_indices]
    
    test_states = states[test_indices]
    test_future_states = future_states[test_indices]
    test_action_sequences = action_sequences[test_indices]
    
    # Create data loaders
    train_subgoal_dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.FloatTensor(train_future_states)
    )
    test_subgoal_dataset = TensorDataset(
        torch.FloatTensor(test_states),
        torch.FloatTensor(test_future_states)
    )
    
    train_policy_dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.FloatTensor(train_future_states),
        torch.FloatTensor(train_action_sequences)
    )
    test_policy_dataset = TensorDataset(
        torch.FloatTensor(test_states),
        torch.FloatTensor(test_future_states),
        torch.FloatTensor(test_action_sequences)
    )
    
    train_subgoal_loader = DataLoader(train_subgoal_dataset, batch_size=batch_size, shuffle=True)
    test_subgoal_loader = DataLoader(test_subgoal_dataset, batch_size=batch_size, shuffle=False)
    
    train_policy_loader = DataLoader(train_policy_dataset, batch_size=batch_size, shuffle=True)
    test_policy_loader = DataLoader(test_policy_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the hierarchical BC model
    state_dim = states.shape[1]  # Should be 2
    action_dim = action_sequences.shape[2]  # Should be 2
    
    model = HierarchicalBC(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        device=device
    )
    
    # Create optimizers
    subgoal_optimizer = optim.Adam(model.subgoal_predictor.parameters(), lr=learning_rate)
    policy_optimizer = optim.Adam(model.subgoal_policy.parameters(), lr=learning_rate)
    
    # Training history
    subgoal_train_losses = []
    subgoal_test_losses = []
    policy_train_losses = []
    policy_test_losses = []
    
    # First, train the subgoal predictor
    print("Training subgoal predictor...")
    for epoch in range(subgoal_epochs):
        # Training
        model.train_mode()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        for states_batch, future_states_batch in train_subgoal_loader:
            # Move data to device
            states_batch = states_batch.to(device)
            future_states_batch = future_states_batch.to(device)
            
            # Zero gradients
            subgoal_optimizer.zero_grad()
            
            # Compute loss
            loss, recon_loss, kl_loss = model.compute_subgoal_loss(states_batch, future_states_batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            subgoal_optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
        
        # Calculate average losses
        train_loss /= len(train_subgoal_loader)
        train_recon_loss /= len(train_subgoal_loader)
        train_kl_loss /= len(train_subgoal_loader)
        
        # Testing
        model.eval_mode()
        test_loss = 0.0
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        
        with torch.no_grad():
            for states_batch, future_states_batch in test_subgoal_loader:
                # Move data to device
                states_batch = states_batch.to(device)
                future_states_batch = future_states_batch.to(device)
                
                # Compute loss
                loss, recon_loss, kl_loss = model.compute_subgoal_loss(states_batch, future_states_batch)
                
                # Accumulate losses
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()
        
        # Calculate average losses
        test_loss /= len(test_subgoal_loader)
        test_recon_loss /= len(test_subgoal_loader)
        test_kl_loss /= len(test_subgoal_loader)
        
        # Save losses
        subgoal_train_losses.append(train_loss)
        subgoal_test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{subgoal_epochs}, "
                  f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}), "
                  f"Test Loss: {test_loss:.4f} (Recon: {test_recon_loss:.4f}, KL: {test_kl_loss:.4f})")
    
    # Save the trained subgoal predictor
    subgoal_model_path = os.path.join(model_save_path, 'subgoal_predictor.pt')
    torch.save(model.subgoal_predictor.state_dict(), subgoal_model_path)
    print(f"Subgoal predictor saved to {subgoal_model_path}")
    
    # Now, train the low-level policy
    print("\nTraining low-level policy...")
    for epoch in range(policy_epochs):
        # Training
        model.train_mode()
        train_loss = 0.0
        
        for states_batch, future_states_batch, action_sequences_batch in train_policy_loader:
            # Move data to device
            states_batch = states_batch.to(device)
            future_states_batch = future_states_batch.to(device)
            action_sequences_batch = action_sequences_batch.to(device)
            
            # Zero gradients
            policy_optimizer.zero_grad()
            
            # Compute loss
            loss = model.compute_policy_loss(states_batch, future_states_batch, action_sequences_batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            policy_optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
        
        # Calculate average loss
        train_loss /= len(train_policy_loader)
        
        # Testing
        model.eval_mode()
        test_loss = 0.0
        
        with torch.no_grad():
            for states_batch, future_states_batch, action_sequences_batch in test_policy_loader:
                # Move data to device
                states_batch = states_batch.to(device)
                future_states_batch = future_states_batch.to(device)
                action_sequences_batch = action_sequences_batch.to(device)
                
                # Compute loss
                loss = model.compute_policy_loss(states_batch, future_states_batch, action_sequences_batch)
                
                # Accumulate loss
                test_loss += loss.item()
        
        # Calculate average loss
        test_loss /= len(test_policy_loader)
        
        # Save losses
        policy_train_losses.append(train_loss)
        policy_test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{policy_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Save the trained policy
    policy_model_path = os.path.join(model_save_path, 'subgoal_policy.pt')
    torch.save(model.subgoal_policy.state_dict(), policy_model_path)
    print(f"Subgoal policy saved to {policy_model_path}")
    
    # Plot training curves
    plot_training_curves(
        subgoal_train_losses, subgoal_test_losses,
        policy_train_losses, policy_test_losses,
        model_save_path
    )
    
    return model

def train_hierarchical_bc_with_gaze(
    data_path='data/hierarchical_data_with_gaze.npz', 
    model_save_path='models',
    batch_size=64,
    subgoal_epochs=100,
    policy_epochs=100,
    learning_rate=1e-3,
    latent_dim=2,
    hidden_dim=64,
    sequence_length=5,
    seed=42
):
    """
    Train a hierarchical BC model with gaze information on trajectory data.
    
    Args:
        data_path: Path to the processed hierarchical data with gaze
        model_save_path: Directory to save the trained models
        batch_size: Batch size for training
        subgoal_epochs: Number of epochs to train the subgoal predictor
        policy_epochs: Number of epochs to train the low-level policy
        learning_rate: Learning rate for optimizers
        latent_dim: Dimension of the latent space in the CVAE
        hidden_dim: Hidden layer dimension
        sequence_length: Length of action sequences
        seed: Random seed
    """
    # This function is similar to the above, but uses gaze data to condition
    # the subgoal prediction during training.
    # In this implementation, we'll use the gaze data directly as the subgoal
    # rather than training a complex CVAE that takes gaze as input.
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model save directory
    os.makedirs(model_save_path, exist_ok=True)
    
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the dataset
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
        states = data['states']
        gazes = data['gazes']
        future_states = data['future_states']
        action_sequences = data['action_sequences']
    else:
        print(f"Data file not found. Please run prepare_hierarchical_data.py first.")
        return None
    
    # Create train/test split
    n_samples = len(states)
    n_test = int(n_samples * 0.2)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    train_states = states[train_indices]
    train_gazes = gazes[train_indices]
    train_future_states = future_states[train_indices]
    train_action_sequences = action_sequences[train_indices]
    
    test_states = states[test_indices]
    test_gazes = gazes[test_indices]
    test_future_states = future_states[test_indices]
    test_action_sequences = action_sequences[test_indices]
    
    # For the gaze-conditioned model, we'll skip training the subgoal predictor
    # and directly use the gaze as the subgoal
    
    # Create data loader for policy training
    train_policy_dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.FloatTensor(train_gazes),  # Using gaze as subgoal
        torch.FloatTensor(train_action_sequences)
    )
    test_policy_dataset = TensorDataset(
        torch.FloatTensor(test_states),
        torch.FloatTensor(test_gazes),  # Using gaze as subgoal
        torch.FloatTensor(test_action_sequences)
    )
    
    train_policy_loader = DataLoader(train_policy_dataset, batch_size=batch_size, shuffle=True)
    test_policy_loader = DataLoader(test_policy_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the hierarchical BC model
    state_dim = states.shape[1]  # Should be 2
    action_dim = action_sequences.shape[2]  # Should be 2
    
    model = HierarchicalBC(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        device=device
    )
    
    # Create optimizer for the policy
    policy_optimizer = optim.Adam(model.subgoal_policy.parameters(), lr=learning_rate)
    
    # Training history
    policy_train_losses = []
    policy_test_losses = []
    
    # Train the low-level policy using gaze as subgoal
    print("Training low-level policy with gaze...")
    for epoch in range(policy_epochs):
        # Training
        model.train_mode()
        train_loss = 0.0
        
        for states_batch, gazes_batch, action_sequences_batch in train_policy_loader:
            # Move data to device
            states_batch = states_batch.to(device)
            gazes_batch = gazes_batch.to(device)  # Gaze as subgoal
            action_sequences_batch = action_sequences_batch.to(device)
            
            # Zero gradients
            policy_optimizer.zero_grad()
            
            # Compute loss
            loss = model.compute_policy_loss(states_batch, gazes_batch, action_sequences_batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            policy_optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
        
        # Calculate average loss
        train_loss /= len(train_policy_loader)
        
        # Testing
        model.eval_mode()
        test_loss = 0.0
        
        with torch.no_grad():
            for states_batch, gazes_batch, action_sequences_batch in test_policy_loader:
                # Move data to device
                states_batch = states_batch.to(device)
                gazes_batch = gazes_batch.to(device)  # Gaze as subgoal
                action_sequences_batch = action_sequences_batch.to(device)
                
                # Compute loss
                loss = model.compute_policy_loss(states_batch, gazes_batch, action_sequences_batch)
                
                # Accumulate loss
                test_loss += loss.item()
        
        # Calculate average loss
        test_loss /= len(test_policy_loader)
        
        # Save losses
        policy_train_losses.append(train_loss)
        policy_test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{policy_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Save the trained policy
    policy_model_path = os.path.join(model_save_path, 'subgoal_policy_with_gaze.pt')
    torch.save(model.subgoal_policy.state_dict(), policy_model_path)
    print(f"Subgoal policy with gaze saved to {policy_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(policy_train_losses, label='Train Loss')
    plt.plot(policy_test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Policy Training with Gaze as Subgoal')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_save_path, 'policy_with_gaze_training_curve.png'))
    plt.close()
    
    return model

def plot_training_curves(subgoal_train_losses, subgoal_test_losses, 
                         policy_train_losses, policy_test_losses, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(15, 5))
    
    # Plot subgoal predictor training curve
    plt.subplot(1, 2, 1)
    plt.plot(subgoal_train_losses, label='Train Loss')
    plt.plot(subgoal_test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Subgoal Predictor Training')
    plt.legend()
    plt.grid(True)
    
    # Plot policy training curve
    plt.subplot(1, 2, 2)
    plt.plot(policy_train_losses, label='Train Loss')
    plt.plot(policy_test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Low-Level Policy Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'hierarchical_bc_training_curves.png'))
    plt.close()

if __name__ == "__main__":
    # Train hierarchical BC model
    print("Training hierarchical BC model without gaze...")
    model = train_hierarchical_bc(
        subgoal_epochs=50,
        policy_epochs=50
    )
    
    print("\nTraining hierarchical BC model with gaze...")
    model_with_gaze = train_hierarchical_bc_with_gaze(
        policy_epochs=50
    ) 