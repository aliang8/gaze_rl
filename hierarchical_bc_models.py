import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SubgoalPredictorCVAE(nn.Module):
    """
    Conditional Variational Autoencoder for predicting subgoals.
    The model conditions on the current state and predicts a distribution
    over possible future states (subgoals).
    """
    
    def __init__(self, state_dim=2, latent_dim=2, hidden_dim=64):
        super(SubgoalPredictorCVAE, self).__init__()
        
        # Encoder: q(z|s_t, sg)
        self.encoder_fc1 = nn.Linear(state_dim * 2, hidden_dim)  # Takes (state, subgoal) pair
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: p(sg|s_t, z)
        self.decoder_fc1 = nn.Linear(state_dim + latent_dim, hidden_dim)  # Takes (state, z) pair
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, state_dim)
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
    def encode(self, state, subgoal):
        """
        Encode state and subgoal into latent distribution parameters.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            subgoal: Subgoal state tensor (batch_size, state_dim)
            
        Returns:
            mean: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        x = torch.cat([state, subgoal], dim=1)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mean: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
            
        Returns:
            z: Sampled latent vector (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, state, z):
        """
        Decode state and latent code into a predicted subgoal.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            z: Latent code tensor (batch_size, latent_dim)
            
        Returns:
            subgoal: Predicted subgoal state (batch_size, state_dim)
        """
        x = torch.cat([state, z], dim=1)
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        subgoal = self.decoder_out(x)
        return subgoal
    
    def forward(self, state, subgoal):
        """
        Forward pass through the CVAE.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            subgoal: Subgoal state tensor (batch_size, state_dim)
            
        Returns:
            subgoal_recon: Reconstructed subgoal (batch_size, state_dim)
            mean: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        mean, logvar = self.encode(state, subgoal)
        z = self.reparameterize(mean, logvar)
        subgoal_recon = self.decode(state, z)
        return subgoal_recon, mean, logvar
    
    def sample_subgoal(self, state, num_samples=1):
        """
        Sample subgoals given the current state.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            num_samples: Number of subgoals to sample for each state
            
        Returns:
            subgoals: Sampled subgoals (batch_size, num_samples, state_dim)
        """
        batch_size = state.size(0)
        
        # Sample latent codes from prior distribution
        z = torch.randn(batch_size, num_samples, self.latent_dim, device=state.device)
        
        # Expand state to match z dimensions
        expanded_state = state.unsqueeze(1).expand(-1, num_samples, -1)
        
        # Reshape for batch processing
        reshaped_state = expanded_state.reshape(batch_size * num_samples, self.state_dim)
        reshaped_z = z.reshape(batch_size * num_samples, self.latent_dim)
        
        # Decode
        subgoals = self.decode(reshaped_state, reshaped_z)
        
        # Reshape back
        subgoals = subgoals.reshape(batch_size, num_samples, self.state_dim)
        
        return subgoals


class SubgoalReachingPolicy(nn.Module):
    """
    LSTM-based policy for reaching subgoals.
    Given the current state and a target subgoal, it predicts
    a sequence of actions to reach the subgoal.
    """
    
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=64, sequence_length=5):
        super(SubgoalReachingPolicy, self).__init__()
        
        # Input processing layers
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        self.subgoal_fc = nn.Linear(state_dim, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Processed state and subgoal
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output layers to predict actions
        self.action_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.action_fc2 = nn.Linear(hidden_dim, action_dim)
        
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
    
    def forward(self, state, subgoal, hidden=None):
        """
        Forward pass to predict actions.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            subgoal: Target subgoal tensor (batch_size, state_dim)
            hidden: Initial hidden state for LSTM (optional)
            
        Returns:
            actions: Predicted actions (batch_size, sequence_length, action_dim)
            final_hidden: Final hidden state
        """
        batch_size = state.size(0)
        
        # Process state and subgoal
        state_feat = F.relu(self.state_fc(state))
        subgoal_feat = F.relu(self.subgoal_fc(subgoal))
        
        # Combine features
        combined_feat = torch.cat([state_feat, subgoal_feat], dim=1)
        
        # Expand for sequence prediction
        lstm_input = combined_feat.unsqueeze(1).expand(-1, self.sequence_length, -1)
        
        # LSTM forward pass
        if hidden is None:
            lstm_out, final_hidden = self.lstm(lstm_input)
        else:
            lstm_out, final_hidden = self.lstm(lstm_input, hidden)
        
        # Predict actions
        x = F.relu(self.action_fc1(lstm_out))
        actions = self.action_fc2(x)
        
        # Normalize actions to unit norm
        action_norms = torch.norm(actions, dim=2, keepdim=True)
        action_norms = torch.clamp(action_norms, min=1e-8)
        normalized_actions = actions / action_norms
        
        return normalized_actions, final_hidden
    
    def predict_single_action(self, state, subgoal, hidden=None):
        """
        Predict a single action (first in the sequence) for the current state.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            subgoal: Target subgoal tensor (batch_size, state_dim)
            hidden: Initial hidden state for LSTM (optional)
            
        Returns:
            action: First predicted action (batch_size, action_dim)
            hidden: Updated hidden state
        """
        # Get full sequence of actions
        actions, hidden = self.forward(state, subgoal, hidden)
        
        # Return only the first action
        return actions[:, 0, :], hidden


class HierarchicalBC:
    """
    Hierarchical Behavioral Cloning model that combines a subgoal predictor
    and a subgoal-reaching policy.
    """
    
    def __init__(self, state_dim=2, action_dim=2, latent_dim=2, hidden_dim=64, 
                 sequence_length=5, device='cpu'):
        self.subgoal_predictor = SubgoalPredictorCVAE(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.subgoal_policy = SubgoalReachingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length
        ).to(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.device = device
        
        # For tracking active subgoals during rollout
        self.current_subgoal = None
        self.steps_to_next_subgoal = 0
        self.lstm_hidden = None
    
    def compute_subgoal_loss(self, states, subgoals):
        """
        Compute CVAE loss for the subgoal predictor.
        
        Args:
            states: Current states tensor (batch_size, state_dim)
            subgoals: Target subgoals tensor (batch_size, state_dim)
            
        Returns:
            loss: CVAE loss (reconstruction + KL divergence)
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence loss component
        """
        # Forward pass through the CVAE
        subgoals_recon, mean, logvar = self.subgoal_predictor(states, subgoals)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(subgoals_recon, subgoals)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / states.size(0)  # Normalize by batch size
        
        # Total loss
        loss = recon_loss + 0.1 * kl_loss  # Beta-VAE with beta=0.1
        
        return loss, recon_loss, kl_loss
    
    def compute_policy_loss(self, states, subgoals, action_sequences):
        """
        Compute behavioral cloning loss for the subgoal-reaching policy.
        
        Args:
            states: Current states tensor (batch_size, state_dim)
            subgoals: Target subgoals tensor (batch_size, state_dim)
            action_sequences: Ground truth action sequences (batch_size, sequence_length, action_dim)
            
        Returns:
            loss: Behavioral cloning loss (MSE)
        """
        # Forward pass to get predicted actions
        pred_actions, _ = self.subgoal_policy(states, subgoals)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_actions, action_sequences)
        
        return loss
    
    def train_mode(self):
        """Set both models to training mode."""
        self.subgoal_predictor.train()
        self.subgoal_policy.train()
    
    def eval_mode(self):
        """Set both models to evaluation mode."""
        self.subgoal_predictor.eval()
        self.subgoal_policy.eval()
    
    def predict_action(self, state, gaze=None, reset=False):
        """
        Predict an action given the current state and optionally gaze data.
        Uses hierarchical approach:
        1. If it's time for a new subgoal, predict one
        2. Use low-level policy to move toward current subgoal
        
        Args:
            state: Current state tensor (1, state_dim)
            gaze: Gaze position tensor (1, 2) or None
            reset: Whether to reset the internal state (start of a new episode)
            
        Returns:
            action: Predicted action
        """
        # Convert to tensor if needed
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if reset or self.current_subgoal is None or self.steps_to_next_subgoal <= 0:
            # Time to predict a new subgoal
            with torch.no_grad():
                # Sample latent code - if we have gaze, we can use it to condition the sampling
                if gaze is not None and torch.is_tensor(gaze):
                    # If we have gaze, we can use it as a hint for which goal is active
                    # We'll directly use it as a subgoal
                    self.current_subgoal = gaze
                else:
                    # Sample from the prior
                    subgoals = self.subgoal_predictor.sample_subgoal(state)
                    self.current_subgoal = subgoals[:, 0, :]  # Take the first sample
            
            # Reset subgoal counter and LSTM hidden state
            self.steps_to_next_subgoal = self.sequence_length
            self.lstm_hidden = None
        
        # Predict action from low-level policy
        with torch.no_grad():
            action, self.lstm_hidden = self.subgoal_policy.predict_single_action(
                state, self.current_subgoal, self.lstm_hidden
            )
        
        # Decrement counter
        self.steps_to_next_subgoal -= 1
        
        return action.squeeze(0).cpu().numpy()
    
    def save_models(self, subgoal_path, policy_path):
        """Save both models to disk."""
        torch.save(self.subgoal_predictor.state_dict(), subgoal_path)
        torch.save(self.subgoal_policy.state_dict(), policy_path)
    
    def load_models(self, subgoal_path, policy_path):
        """Load both models from disk."""
        self.subgoal_predictor.load_state_dict(torch.load(subgoal_path))
        self.subgoal_policy.load_state_dict(torch.load(policy_path))
    
    def predict_subgoal(self, state):
        """
        Predict a subgoal given the current state by sampling from the CVAE.
        
        Args:
            state: Current state tensor (batch_size, state_dim)
            
        Returns:
            subgoal: Predicted subgoal (batch_size, state_dim)
        """
        with torch.no_grad():
            # Sample from the prior distribution
            subgoal = self.subgoal_predictor.sample_subgoal(state)
            # Take the first sample from each batch
            subgoal = subgoal[:, 0, :]
        return subgoal 