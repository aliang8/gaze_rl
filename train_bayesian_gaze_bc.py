import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from hierarchical_bc_models import SubgoalPredictorCVAE, SubgoalReachingPolicy
from bayesian_gaze_model import GazeLikelihoodModel
import matplotlib.pyplot as plt


def load_hierarchical_models(state_dim=2, action_dim=2, device='cpu'):
    """
    Load the pretrained CVAE and subgoal-reaching policy models
    
    Args:
        state_dim: Dimension of the state
        action_dim: Dimension of the action
        device: Device to load the models on
        
    Returns:
        cvae: Pretrained CVAE model
        subgoal_policy: Pretrained subgoal-reaching policy
    """
    # Create model instances
    cvae = SubgoalPredictorCVAE(state_dim=state_dim, latent_dim=2, hidden_dim=64)
    subgoal_policy = SubgoalReachingPolicy(state_dim=state_dim, 
                                           action_dim=action_dim, hidden_dim=64, 
                                           sequence_length=5)
    
    # Load pretrained models
    cvae.load_state_dict(torch.load('models/subgoal_predictor.pt', map_location=device))
    subgoal_policy.load_state_dict(torch.load('models/subgoal_policy.pt', map_location=device))
    
    # Set models to evaluation mode
    cvae.eval()
    subgoal_policy.eval()
    
    return cvae, subgoal_policy


def prepare_data(data_path='data/hierarchical_dataset_with_gaze.npz', device='cpu'):
    """
    Prepare data for training the gaze likelihood model
    
    Args:
        data_path: Path to the hierarchical dataset with gaze
        device: Device to load the data on
        
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    # Load data
    data = np.load(data_path)
    states = torch.FloatTensor(data['states']).to(device)
    future_states = torch.FloatTensor(data['future_states']).to(device)  # These are our subgoals
    gazes = torch.FloatTensor(data['gazes']).to(device)
    
    # Create dataset
    dataset = TensorDataset(states, future_states, gazes)
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


def train_gaze_likelihood_model(model, train_loader, test_loader, num_epochs=100, lr=0.001, device='cpu'):
    """
    Train the gaze likelihood model
    
    Args:
        model: GazeLikelihoodModel instance
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        train_losses: List of training losses
        test_losses: List of testing losses
    """
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store losses
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for states, subgoals, gazes in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = model.loss(gazes, states, subgoals)
            
            # Backward pass and update
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
        
        # Average training loss
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Testing
        model.eval()
        epoch_test_loss = 0.0
        
        with torch.no_grad():
            for states, subgoals, gazes in test_loader:
                # Compute loss
                loss = model.loss(gazes, states, subgoals)
                
                # Accumulate loss
                epoch_test_loss += loss.item()
        
        # Average testing loss
        epoch_test_loss /= len(test_loader)
        test_losses.append(epoch_test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")
    
    return train_losses, test_losses


def visualize_training(train_losses, test_losses, save_path='results/hierarchical/bayesian'):
    """
    Visualize training progress
    
    Args:
        train_losses: List of training losses
        test_losses: List of testing losses
        save_path: Directory to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Gaze Likelihood Model Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'gaze_likelihood_training.png'))
    plt.close()


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load pretrained hierarchical models
    cvae, subgoal_policy = load_hierarchical_models(state_dim=args.state_dim, action_dim=args.action_dim, device=device)
    
    # Prepare data
    train_loader, test_loader = prepare_data(data_path=args.data_path, device=device)
    
    # Create gaze likelihood model
    gaze_model = GazeLikelihoodModel(state_dim=args.state_dim, subgoal_dim=args.state_dim, hidden_dim=args.hidden_dim).to(device)
    
    # Train the model
    train_losses, test_losses = train_gaze_likelihood_model(
        model=gaze_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Visualize training progress
    visualize_training(train_losses, test_losses, save_path=args.save_dir)
    
    # Save the model
    torch.save(gaze_model.state_dict(), os.path.join(args.save_dir, 'gaze_likelihood_model.pt'))
    print(f"Model saved to {os.path.join(args.save_dir, 'gaze_likelihood_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Bayesian gaze likelihood model")
    parser.add_argument("--data-path", type=str, default="data/hierarchical_data_with_gaze.npz",
                        help="Path to the hierarchical dataset with gaze")
    parser.add_argument("--save-dir", type=str, default="results/hierarchical/bayesian",
                        help="Directory to save the model and results")
    parser.add_argument("--state-dim", type=int, default=2,
                        help="Dimension of the state space")
    parser.add_argument("--action-dim", type=int, default=2,
                        help="Dimension of the action space")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Dimension of the hidden layers")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    main(args) 