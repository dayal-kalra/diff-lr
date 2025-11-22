import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Dict

class LRPredictor(nn.Module):
    """Predicts learning rate from training features using GRU + MLP"""
    
    gru_hidden_size: int = 32
    mlp_hidden_size: int = 64
    lr_min: float = 1e-8
    lr_max: float = 100.0
    
    @nn.compact
    def __call__(self, features: Dict):
        """
        Args:
            features: dict with
                - 'weight_norm': scalar
                - 'grad_norm': scalar  
                - 'loss_current': scalar
                - 'step_progress': scalar (t/T)
                - 'loss_history': array[100]
        Returns:
            lr: scalar in [lr_min, lr_max]
        """
        # Process loss history with GRU
        # Reshape to (seq_len, 1) for GRU
        loss_seq = features['loss_history'][:, None]  # (100, 1)
        
        # GRU cell
        gru_cell = nn.GRUCell(features=self.gru_hidden_size)
        
        # Process sequence
        carry = gru_cell.initialize_carry(jax.random.PRNGKey(0), (self.gru_hidden_size,))
        for i in range(loss_seq.shape[0]):
            carry, _ = gru_cell(carry, loss_seq[i])
        
        gru_output = carry  # (gru_hidden_size,)
        
        # Concatenate with scalar features
        scalar_features = jnp.array([
            features['weight_norm'],
            features['grad_norm'],
            features['loss_current'],
            features['step_progress']
        ])
        
        x = jnp.concatenate([scalar_features, gru_output])  # (4 + 32,)
        
        # MLP
        x = nn.Dense(self.mlp_hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.mlp_hidden_size)(x)
        x = nn.relu(x)
        
        # Output logit
        logit = nn.Dense(1)(x)
        logit = logit.squeeze()
        
        # Map to [lr_min, lr_max]
        lr = nn.sigmoid(logit) * (self.lr_max - self.lr_min) + self.lr_min

        return jnp.asarray(lr)


def compute_features(params, grads, loss, loss_history, step, total_steps):
    """Compute features for LR predictor"""
    
    # Weight norm
    weight_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(params)))
    
    # Gradient norm
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    
    # Step progress
    step_progress = step / total_steps
    
    features = {
        'weight_norm': weight_norm,
        'grad_norm': grad_norm,
        'loss_current': loss,
        'step_progress': step_progress,
        'loss_history': loss_history  # array of last 100 losses
    }
    
    return features