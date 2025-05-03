import torch
import torch.nn as nn
import torch.nn.functional as F

class BCPolicy(nn.Module):
    """
    Behavioral Cloning Policy Network without gaze information.
    Maps state observations to continuous action vectors.
    """
    
    def __init__(self, state_dim=2, hidden_dim=64, action_dim=2):
        super(BCPolicy, self).__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            action: Tensor of shape (batch_size, action_dim) 
                  representing continuous action vector
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Output unnormalized action vector
        action = self.fc3(x)
        
        # Normalize action to have magnitude <= 1.0
        action_norm = torch.norm(action, dim=1, keepdim=True)
        # Avoid division by zero
        action_norm = torch.clamp(action_norm, min=1e-8)
        
        # Normalize action
        normalized_action = action / action_norm
        
        return normalized_action

class BCPolicyWithGaze(nn.Module):
    """
    Behavioral Cloning Policy Network with gaze information.
    Maps state observations and gaze information to continuous action vectors.
    """
    
    def __init__(self, state_dim=2, gaze_dim=2, hidden_dim=64, action_dim=2):
        super(BCPolicyWithGaze, self).__init__()
        
        # Define network layers
        self.fc1 = nn.Linear(state_dim + gaze_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state, gaze):
        """
        Forward pass through the network.
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            gaze: Tensor of shape (batch_size, gaze_dim)
        
        Returns:
            action: Tensor of shape (batch_size, action_dim) 
                  representing continuous action vector
        """
        # Concatenate state and gaze information
        x = torch.cat([state, gaze], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output unnormalized action vector
        action = self.fc3(x)
        
        # Normalize action to have magnitude <= 1.0
        action_norm = torch.norm(action, dim=1, keepdim=True)
        # Avoid division by zero
        action_norm = torch.clamp(action_norm, min=1e-8)
        
        # Normalize action
        normalized_action = action / action_norm
        
        return normalized_action 