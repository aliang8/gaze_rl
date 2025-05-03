import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hierarchical_bc_models import SubgoalPredictorCVAE, SubgoalReachingPolicy, HierarchicalBC

class GazeLikelihoodModel(nn.Module):
    """
    Model for p(gaze_t | subgoal, s_t), the likelihood of observing gaze given a state and subgoal
    This model will be used in Bayesian inference to weight the prior distribution from CVAE
    """
    def __init__(self, state_dim, subgoal_dim, hidden_dim=128):
        super(GazeLikelihoodModel, self).__init__()
        self.state_dim = state_dim
        self.subgoal_dim = subgoal_dim
        
        # Network for predicting the parameters of the gaze distribution
        self.fc1 = nn.Linear(state_dim + subgoal_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, 2)  # Predicting mean of gaze (x, y)
        self.fc_logvar = nn.Linear(hidden_dim, 2)  # Predicting log variance of gaze (x, y)
        
    def forward(self, state, subgoal):
        """
        Forward pass of the gaze likelihood model
        
        Args:
            state: Current state [batch_size, state_dim]
            subgoal: Potential subgoal [batch_size, subgoal_dim]
            
        Returns:
            mu: Mean of predicted gaze [batch_size, 2]
            logvar: Log variance of predicted gaze [batch_size, 2]
        """
        x = torch.cat([state, subgoal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def log_likelihood(self, gaze, state, subgoal):
        """
        Compute log likelihood of p(gaze | subgoal, state)
        
        Args:
            gaze: Observed gaze [batch_size, 2]
            state: Current state [batch_size, state_dim]
            subgoal: Potential subgoal [batch_size, subgoal_dim]
            
        Returns:
            log_likelihood: Log likelihood of gaze given state and subgoal [batch_size]
        """
        mu, logvar = self.forward(state, subgoal)
        var = torch.exp(logvar)
        
        # Compute the negative log likelihood (Gaussian)
        # log p(gaze | subgoal, state) = -0.5 * [(gaze - mu)^2/var + log(var) + log(2π)]
        log_likelihood = -0.5 * (((gaze - mu) ** 2) / var + logvar + torch.log(torch.tensor(2 * np.pi)))
        
        # Sum over the x, y dimensions
        log_likelihood = log_likelihood.sum(dim=1)
        return log_likelihood
    
    def loss(self, gaze, state, subgoal):
        """
        Compute negative log likelihood loss for training
        
        Args:
            gaze: Observed gaze [batch_size, 2]
            state: Current state [batch_size, state_dim]
            subgoal: Potential subgoal [batch_size, subgoal_dim]
            
        Returns:
            loss: Negative log likelihood loss
        """
        log_likelihood = self.log_likelihood(gaze, state, subgoal)
        return -log_likelihood.mean()


class BayesianGazeBC(HierarchicalBC):
    """
    Bayesian approach to incorporate gaze information with hierarchical BC
    Uses Bayes' rule to compute p(subgoal | state, gaze) ∝ p(gaze | subgoal, state) × p(subgoal | state)
    """
    def __init__(self, state_dim=2, action_dim=2, latent_dim=2, hidden_dim=64, 
                 sequence_length=5, device='cpu', num_samples=10, uncertainty_threshold=0.05):
        super(BayesianGazeBC, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            device=device
        )
        
        # Create gaze likelihood model
        self.gaze_likelihood_model = GazeLikelihoodModel(
            state_dim=state_dim,
            subgoal_dim=state_dim,  # Subgoal is same dimension as state
            hidden_dim=hidden_dim
        ).to(device)
        
        # Number of samples to use for approximating the posterior
        self.num_samples = num_samples
        
        # Threshold for when to use gaze based on subgoal prediction uncertainty
        self.uncertainty_threshold = uncertainty_threshold
        
        # For tracking last computed uncertainty
        self.last_uncertainty = None
        
        # For visualization: store all sampled subgoals and which one was selected
        self.last_sampled_subgoals = None
        self.last_selected_idx = None
    
    def sample_subgoals(self, state, n_samples=10):
        """
        Sample multiple subgoals from the CVAE prior
        
        Args:
            state: Current state [batch_size, state_dim]
            n_samples: Number of subgoals to sample
            
        Returns:
            subgoals: Sampled subgoals [batch_size, n_samples, subgoal_dim]
        """
        batch_size = state.shape[0]
        subgoals = []
        
        for _ in range(n_samples):
            # Sample z from standard normal
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            # Decode to get a subgoal
            subgoal = self.subgoal_predictor.decode(state, z)
            subgoals.append(subgoal.unsqueeze(1))
        
        return torch.cat(subgoals, dim=1)
    
    def calculate_subgoal_uncertainty(self, subgoals):
        """
        Calculate uncertainty across sampled subgoals
        
        Args:
            subgoals: Sampled subgoals [batch_size, n_samples, subgoal_dim]
            
        Returns:
            uncertainty: Scalar value representing subgoal prediction uncertainty
        """
        # Calculate variance for each dimension, then sum across dimensions
        variance = torch.var(subgoals, dim=1).sum(dim=1)
        # Average across batch
        mean_variance = variance.mean().item()
        return mean_variance
    
    def predict_subgoal_with_uncertainty(self, state, gaze=None, n_samples=None):
        """
        Predict subgoal and calculate uncertainty
        If uncertainty is above threshold and gaze is provided, use gaze for Bayesian inference
        
        Args:
            state: Current state [batch_size, state_dim]
            gaze: Observed gaze [batch_size, 2] or None
            n_samples: Number of subgoals to sample (default: self.num_samples)
            
        Returns:
            best_subgoal: Subgoal with highest posterior probability [batch_size, subgoal_dim]
            used_gaze: Boolean indicating whether gaze was used
        """
        if n_samples is None:
            n_samples = self.num_samples
            
        batch_size = state.shape[0]
        
        # Sample subgoals from prior p(subgoal | state)
        subgoals = self.sample_subgoals(state, n_samples)
        
        # Store for visualization
        self.last_sampled_subgoals = subgoals.detach().cpu().numpy()
        
        # Calculate uncertainty
        uncertainty = self.calculate_subgoal_uncertainty(subgoals)
        self.last_uncertainty = uncertainty
        
        # Choose whether to use gaze based on uncertainty
        used_gaze = False
        if uncertainty > self.uncertainty_threshold and gaze is not None:
            # Uncertainty is high, use gaze if available
            used_gaze = True
            
            # Compute likelihood p(gaze | subgoal, state) for each subgoal
            log_likelihoods = []
            for i in range(n_samples):
                subgoal = subgoals[:, i, :]
                log_likelihood = self.gaze_likelihood_model.log_likelihood(gaze, state, subgoal)
                log_likelihoods.append(log_likelihood.unsqueeze(1))
            
            log_likelihoods = torch.cat(log_likelihoods, dim=1)  # [batch_size, n_samples]
            
            # Find the subgoal with the highest likelihood
            best_indices = torch.argmax(log_likelihoods, dim=1)
            
            # Store the selected index for visualization
            self.last_selected_idx = best_indices.detach().cpu().numpy()
        else:
            # Uncertainty is low or no gaze provided, just average the samples
            mean_subgoal = torch.mean(subgoals, dim=1)
            
            # No specific subgoal was selected, we averaged them all
            self.last_selected_idx = None
            
            # We could create a special index to indicate "mean of all" but that's harder to visualize
            # So we'll return the mean subgoal and the flag that we didn't use gaze
            return mean_subgoal, used_gaze
        
        # Extract the best subgoal for each batch item when using gaze
        best_subgoals = torch.zeros(batch_size, subgoals.shape[2], device=self.device)
        for b in range(batch_size):
            best_idx = best_indices[b]
            best_subgoals[b] = subgoals[b, best_idx]
        
        return best_subgoals, used_gaze
    
    def predict_action(self, state, gaze=None):
        """
        Predict action given current state and optional gaze
        The model will only use gaze when subgoal prediction uncertainty is high
        
        Args:
            state: Current state [batch_size, state_dim]
            gaze: Observed gaze [batch_size, 2] or None
            
        Returns:
            action: Predicted action [batch_size, action_dim]
            or
            (action, used_gaze): If return_gaze_usage is True
        """
        # Predict subgoal using uncertainty-based gaze conditioning
        subgoal, used_gaze = self.predict_subgoal_with_uncertainty(state, gaze)
        
        # Predict action using the subgoal-reaching policy
        result = self.subgoal_policy.predict_single_action(state, subgoal)
        
        # Handle case where predict_single_action returns (action, hidden_state)
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result
        
        # Return action along with whether gaze was used
        return action, used_gaze

    def to(self, device):
        """
        Move models to the specified device
        
        Args:
            device: Device to move the models to
        """
        self.device = device
        self.subgoal_predictor.to(device)
        self.subgoal_policy.to(device)
        self.gaze_likelihood_model.to(device)
        return self
        
    def load(self, cvae_path, gaze_model_path, subgoal_policy_path, device='cpu'):
        """
        Load the models from saved checkpoints
        
        Args:
            cvae_path: Path to the saved CVAE model
            gaze_model_path: Path to the saved gaze likelihood model
            subgoal_policy_path: Path to the saved subgoal-reaching policy
            device: Device to load the models on
        """
        self.subgoal_predictor.load_state_dict(torch.load(cvae_path, map_location=device))
        self.subgoal_policy.load_state_dict(torch.load(subgoal_policy_path, map_location=device))
        self.gaze_likelihood_model.load_state_dict(torch.load(gaze_model_path, map_location=device))
        self.to(device)


def load_bayesian_gaze_bc_model(model_path, state_dim=2, action_dim=2, latent_dim=2, hidden_dim=64, sequence_length=5, uncertainty_threshold=0.05):
    """
    Load a trained Bayesian gaze BC model.
    
    Args:
        model_path: Directory containing the saved model files
        state_dim: State dimension
        action_dim: Action dimension
        latent_dim: Latent dimension for CVAE
        hidden_dim: Hidden dimension for neural networks
        sequence_length: Sequence length for LSTM
        uncertainty_threshold: Threshold for when to use gaze
    
    Returns:
        The loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = BayesianGazeBC(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        device=device,
        uncertainty_threshold=uncertainty_threshold
    )
    
    # Load subgoal predictor if available
    subgoal_predictor_path = os.path.join(model_path, 'subgoal_predictor.pt')
    if os.path.exists(subgoal_predictor_path):
        model.subgoal_predictor.load_state_dict(torch.load(subgoal_predictor_path, map_location=device))
        print(f"Loaded subgoal predictor from {subgoal_predictor_path}")
    
    # Load subgoal policy if available
    policy_path = os.path.join(model_path, 'subgoal_policy.pt')
    if os.path.exists(policy_path):
        model.subgoal_policy.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"Loaded subgoal policy from {policy_path}")
    
    # Load gaze likelihood model if available
    gaze_likelihood_path = os.path.join(model_path, 'gaze_likelihood_model.pt')
    if os.path.exists(gaze_likelihood_path):
        model.gaze_likelihood_model.load_state_dict(torch.load(gaze_likelihood_path, map_location=device))
        print(f"Loaded gaze likelihood model from {gaze_likelihood_path}")
    
    model.eval_mode()
    return model 