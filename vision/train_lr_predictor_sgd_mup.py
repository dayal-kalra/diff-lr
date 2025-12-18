# in use imports
import utils.mup_cnns as model_utils
import utils.train_utils as train_utils
import utils.data_utils as data_utils
import utils.loss_utils as loss_utils
from utils.diff_lr import LRPredictor, compute_features
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

import jax
from jax import numpy as jnp
import optax
from flax import linen as nn

from typing import Tuple
from functools import partial

#usual imports
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import orbax.checkpoint
    
# for deterministic gpu computations
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'


# In get_all_checkpoint_paths, use hardcoded checkpoint hyperparameters:
def get_all_checkpoint_paths(cfg):
    """
    Generate all possible checkpoint paths based on cfg structure
    
    Args:
        cfg: base config (only uses dataset_name, model, width, depth, abc)
    
    Returns:
        list of tuples: (checkpoint_path, seed, lr, step)
    """
    # Hardcoded values for checkpoint diversity
    seeds = [0, 42, 777, 1337, 12345]
    lrs = [0.001, 0.01, 0.1, 1.0]
    steps = [i for i in range(0, 10000, 1000)] + [10000]
    widths = [16, 24, 32] 
    # Hardcoded checkpoint hyperparameters (these were used to CREATE the checkpoints)
    ckpt_hparams = {
        'use_bias': 'False',
        'sgd_seed': 1337,
        'loss_name': 'xent',
        'augment': 'True',
        'opt_name': 'sgd',
        'lr_min_factor': 'inf',
        'warmup_exponent': 1.0,
        'decay_schedule_name': 'cosine',
        'decay_exponent': 1.0,
        'warmup_steps': 2000,
        'num_steps': 10000,
        'batch_size': 128,
        'momentum': 0.0,
        'grad_clip': 1.0,
        'weight_decay': 0.0
    }
    
    checkpoint_paths = []
    for width in widths:
        for seed in seeds:
            for lr in lrs:
                for step in steps:
                    # Build checkpoint path using CHECKPOINT hyperparameters
                    ckpt_path = f'{cfg.ckpt_base_dir}/{cfg.dataset_name}_{cfg.model}_{cfg.abc}_n{cfg.width}_d{cfg.depth}_bias{ckpt_hparams["use_bias"]}_{cfg.act_name}_I{seed}_J{ckpt_hparams["sgd_seed"]}_{ckpt_hparams["loss_name"]}_aug{ckpt_hparams["augment"]}_{ckpt_hparams["opt_name"]}_lr{lr:0.1e}_lr{ckpt_hparams["lr_min_factor"]}_k{ckpt_hparams["warmup_exponent"]}_{ckpt_hparams["decay_schedule_name"]}_p{ckpt_hparams["decay_exponent"]}_Tw{ckpt_hparams["warmup_steps"]}_T{ckpt_hparams["num_steps"]}_B{ckpt_hparams["batch_size"]}_m{ckpt_hparams["momentum"]}_gc{ckpt_hparams["grad_clip"]}_wd{ckpt_hparams["weight_decay"]}/step_{step}'
                
                    if os.path.exists(ckpt_path):
                        checkpoint_paths.append((ckpt_path, width, seed, lr, step))
    
    return checkpoint_paths


def sample_checkpoint(checkpoint_paths):
    """
    Randomly sample one checkpoint from available checkpoints
    
    Args:
        checkpoint_paths: list of checkpoint path tuples
    
    Returns:
        checkpoint_path, seed, lr, step
    """
    idx = np.random.randint(0, len(checkpoint_paths))
    return checkpoint_paths[idx]

"""Meta-training for learned learning rates"""

def inner_loop_train(lr_predictor_params, lr_predictor_apply, checkpoint_info, train_ds, cfg):
    """
    Train a task network using predicted learning rates (inner loop)
    
    Args:
        lr_predictor_params: parameters of LR predictor
        lr_predictor_apply: apply function of LR predictor
        checkpoint_info: tuple of (checkpoint_path, seed, lr, step)
        train_ds: training data
        cfg: config
    
    Returns:
        avg_loss: average loss over rollout
    """
    checkpoint_path, width, seed, lr, start_step = checkpoint_info
    x_train, y_train = train_ds
    
    # Initialize task network
    model = models[cfg.model](width=cfg.width, num_classes=cfg.out_dim, act=cfg.act, varw=cfg.varw)
    
    # Load only the model parameters from checkpoint
    checkpoint_path_obj = Path(checkpoint_path).resolve()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    # Load the full checkpoint first to extract params
    restored_full = checkpointer.restore(checkpoint_path_obj)
    restored_params = restored_full['params']  # Extract only params
    
    # Create fresh optimizer with loaded params
    base_optim = optax.inject_hyperparams(optax.sgd)(
        learning_rate=cfg.lr_init, 
        momentum=cfg.momentum
    )
    
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        base_optim
    )
    
    state = train_utils.TrainState.create(
        apply_fn=model.apply,
        params=restored_params,  # Use restored params
        opt=optim
    )
    
    # Training loop
    train_loader = train_utils.data_stream(cfg.sgd_seed, train_ds, cfg.batch_size, augment=False)
    
    # Initialize loss history with zeros
    loss_history = jnp.zeros(100)
    losses = []
    
    for step in range(cfg.rollout_steps):
        batch = next(train_loader)
        
        # Compute gradients and loss
        grads, loss = train_utils.grads_step(state, batch, cfg.loss_fn)
        
        # Compute features and predict learning rate
        features = compute_features(
            state.params, grads, loss, loss_history,
            start_step + step, cfg.rollout_steps
        )
        lr_step = lr_predictor_apply({'params': lr_predictor_params}, features)
        lr_step = jnp.squeeze(lr_step)
        
        # Update the learning rate in optimizer state
        state.opt_state[-1].hyperparams['learning_rate'] = lr_step
        
        # Apply gradients
        state = state.apply_gradients(grads=grads)
        
        # Update loss history
        loss_history = jnp.roll(loss_history, -1)
        loss_history = loss_history.at[-1].set(loss)

        jax.debug.print("[Inner Loop] Step {s}/{total}, Loss: {l:.4f}, LR: {lr:.6f}", 
                          s=step, total=cfg.rollout_steps, l=loss, lr=lr_step)
        
        losses.append(loss)
    
    # Return average loss
    avg_loss = jnp.mean(jnp.array(losses))
    return avg_loss

def meta_train_step(lr_predictor_state, checkpoint_paths, train_ds, cfg):
    """
    One meta-training step (outer loop)
    
    Args:
        lr_predictor_state: TrainState for LR predictor
        checkpoint_paths: list of available checkpoint paths
        train_ds: training dataset
        cfg: config
    
    Returns:
        updated lr_predictor_state, meta_loss
    """
    
    def meta_loss_fn(lr_predictor_params):
        """Compute meta-loss on a single sampled checkpoint"""
        # Sample one checkpoint
        checkpoint_info = sample_checkpoint(checkpoint_paths)
        
        task_loss = inner_loop_train(
            lr_predictor_params,
            lr_predictor_state.apply_fn,
            checkpoint_info,
            train_ds,
            cfg
        )
        
        return task_loss
    
    # Compute meta-gradients (backprop through inner loop)
    meta_loss, meta_grads = jax.value_and_grad(meta_loss_fn)(lr_predictor_state.params)
    
    # Update LR predictor parameters
    lr_predictor_state = lr_predictor_state.apply_gradients(grads=meta_grads)
    
    return lr_predictor_state, meta_loss


def meta_train(cfg, train_ds):
    """Main meta-training loop"""
    
    print("Initializing LR Predictor...")
    
    # Initialize LR predictor
    lr_predictor = LRPredictor(
        gru_hidden_size=cfg.gru_hidden_size,
        mlp_hidden_size=cfg.mlp_hidden_size,
        lr_min=cfg.lr_min,
        lr_max=cfg.lr_max
    )
    
    # Dummy features for initialization
    dummy_features = {
        'weight_norm': 1.0,
        'grad_norm': 1.0,
        'loss_current': 1.0,
        'step_progress': 0.5,
        'loss_history': jnp.ones(100)
    }
    
    key = jax.random.PRNGKey(cfg.meta_init_seed)
    lr_predictor_params = lr_predictor.init(key, dummy_features)['params']
    
    # Count parameters
    num_params = model_utils.count_parameters(lr_predictor_params)
    print(f'LR Predictor has {num_params} parameters')
    
    # Create optimizer for LR predictor (meta-optimizer)
    meta_optim = optax.adam(cfg.meta_lr)
    
    lr_predictor_state = train_utils.TrainState.create(
        apply_fn=lr_predictor.apply,
        params=lr_predictor_params,
        opt=meta_optim
    )
    
    # Get all available checkpoints
    print("Loading checkpoint paths...")
    checkpoint_paths = get_all_checkpoint_paths(cfg)
    print(f"Found {len(checkpoint_paths)} checkpoints")
    
    # Store meta-training results
    meta_results = []
    
    # Meta-training loop
    print(f"Starting meta-training for {cfg.num_meta_steps} steps...")
    print(f"Rollout length: {cfg.rollout_steps}")
    
    for meta_step in range(cfg.num_meta_steps):
        
        # Meta-training step (samples one checkpoint internally)
        lr_predictor_state, meta_loss = meta_train_step(
            lr_predictor_state, checkpoint_paths, train_ds, cfg
        )
        
        # Log results
        result = np.array([meta_step, meta_loss])
        meta_results.append(result)
        
        print(f"Meta-step {meta_step}/{cfg.num_meta_steps}, Meta-loss: {meta_loss:.4f}")
    
    print("Meta-training complete!")
    
    # Save results
    meta_results = np.asarray(meta_results)
    df_meta = pd.DataFrame(meta_results, columns=['meta_step', 'meta_loss'])
    df_meta.to_csv(cfg.meta_results_path, sep='\t', index=False)
    print(f"Meta-training results saved to {cfg.meta_results_path}")
    
    # Save LR predictor weights
    save_checkpoint(lr_predictor_state, cfg.num_meta_steps, cfg.lr_predictor_ckpt_dir)
    print(f"LR predictor weights saved to {cfg.lr_predictor_ckpt_dir}")
    
    return lr_predictor_state

models = {'Myrtle5': model_utils.Myrtle5, 'Myrtle7': model_utils.Myrtle7, 'Myrtle10': model_utils.Myrtle10}
loss_fns = {'mse': loss_utils.mse_loss, 'xent': loss_utils.cross_entropy_loss}

parser = argparse.ArgumentParser(description='Meta-training parameters for learned LR')

# Dataset parameters
parser.add_argument('--dataset_name', type=str, default='cifar-10')
parser.add_argument('--out_dim', type=int, default=10)

# Model parameters (task network architecture only)
parser.add_argument('--abc', type=str, default='mup')
parser.add_argument('--width', type=int, default=16)
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--act_name', type=str, default='relu')
parser.add_argument('--varw', type=float, default=2.0)

# Inner loop training parameters (for LR predictor training)
parser.add_argument('--rollout_steps', type=int, default=40)
parser.add_argument('--lr_init', type=float, default=0.01)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--sgd_seed', type=int, default=1337)
parser.add_argument('--loss_name', type=str, default='xent')

# Checkpoint directory
parser.add_argument('--ckpt_base_dir', type=str, default='checkpoints/mup')

# Meta-training parameters
parser.add_argument('--num_meta_steps', type=int, default=100)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--meta_init_seed', type=int, default=42)
parser.add_argument('--meta_log_interval', type=int, default=10)

# LR Predictor parameters
parser.add_argument('--gru_hidden_size', type=int, default=32)
parser.add_argument('--mlp_hidden_size', type=int, default=64)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--lr_max', type=float, default=1.0)

# Paths
parser.add_argument('--ds_dir', type=str, default='data/')
parser.add_argument('--save_dir', type=str, default='meta_results')
parser.add_argument('--lr_predictor_ckpt_dir', type=str, default='lr_predictor_checkpoints')

cfg = parser.parse_args()

# Setup
cfg.model = f'Myrtle{cfg.depth}'
cfg.act = model_utils.activations[cfg.act_name]
cfg.loss_fn = loss_fns[cfg.loss_name]

# create save dir
os.makedirs(cfg.save_dir, exist_ok=True)
os.makedirs(cfg.lr_predictor_ckpt_dir, exist_ok=True)

# Load data
print("Loading dataset...")
(x_train, y_train), (x_test, y_test) = data_utils.load_image_data(
    cfg.ds_dir, cfg.dataset_name, flatten=False, subset=False
)

# Standardize
x_train = data_utils._standardize(x_train, abc='sp')
x_test = data_utils._standardize(x_test, abc='sp')

# One-hot encode
y_train = jax.nn.one_hot(y_train, cfg.out_dim)
y_test = jax.nn.one_hot(y_test, cfg.out_dim)

print(f"Dataset: {cfg.dataset_name}")
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# Meta-training results path
cfg.meta_results_path = f'{cfg.save_dir}/meta_train_{cfg.dataset_name}_{cfg.model}_n{cfg.width}_T{cfg.rollout_steps}_meta{cfg.num_meta_steps}.tab'

print(cfg)

# Run meta-training
lr_predictor_state = meta_train(cfg, (x_train, y_train))

print("Done!")
