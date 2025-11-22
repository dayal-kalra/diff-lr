import jax.numpy as jnp
from jax import jit

@jit
def polynomial_warmup(step, init_value, end_value, num_steps, exponent = 1.0):
    " Polynomial warmup "
    rate = step / num_steps
    lr = init_value + (end_value - init_value) * rate ** exponent
    return jnp.minimum(end_value, lr)

@jit
def polynomial_decay(step, init_value, end_value, num_steps, exponent = 1.0):
    " Polynomial decay schedule "
    rate = step / num_steps
    lr = init_value + (end_value - init_value) * rate ** exponent
    return jnp.maximum(end_value, lr)

@jit
def cosine_decay(step, init_value, end_value, num_steps, exponent = 1.0):
    """ Cosine decay schedule """
    cosine_decay = 0.5 * (1 + jnp.cos( jnp.pi * step / num_steps))
    return end_value + (init_value - end_value) * cosine_decay ** exponent

def extract_learning_rate(opt_state):

    """Extract the current learning rate from the optimizer state"""
    # If using a chained optimizer, the base optimizer state is in the second element
    if isinstance(opt_state, tuple) and len(opt_state) > 1:
        # Access the hyperparams in the last component (SGD)
        if hasattr(opt_state[-1], 'hyperparams') and 'learning_rate' in opt_state[-1].hyperparams:
            return opt_state[-1].hyperparams['learning_rate']
    # If using a single optimizer
    elif hasattr(opt_state, 'hyperparams') and 'learning_rate' in opt_state.hyperparams:
        return opt_state.hyperparams['learning_rate']

    # If we can't extract it, return None
    print(f'Cannot extract the learning rate')
    return None

# NOTE: Not in use; only for reference

def create_lr_schedule(cfg):
    # Create a warmup schedule
    warmup_schedule = optax.linear_schedule(
        init_value = cfg.lr_init,
        end_value = cfg.lr_peak,
        transition_steps = cfg.warmup_steps
    )

    # Create a cosine decay schedule
    cosine_schedule = optax.cosine_decay_schedule(
        init_value = cfg.lr_peak,
        decay_steps = cfg.num_steps - cfg.warmup_steps,
        alpha = 1/10.
    )

    # Combine them
    combined_schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[cfg.warmup_steps]
    )

    return combined_schedule

