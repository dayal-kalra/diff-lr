from typing import NamedTuple
import jax.flatten_util
import jax.numpy as jnp
import optax
import jax
import flax

def flatten_pytree(pytree, prefix = ''):
    flat_dict = {}
    for key, value in pytree.items():
        # Construct the new key path
        new_key = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            # If the value is a dictionary, recurse further
            flat_dict.update(flatten_pytree(value, new_key))
        else:
            # Otherwise, store the value with its accumulated key path
            flat_dict[new_key] = value
    return flat_dict

class SGDState(NamedTuple):
    mu: optax.Updates  # moving average of gradients
    count: jnp.ndarray  # Timestep

def SGD(lr_pytree, learning_rate: float = 1.0, momentum: float = 0.0, weight_decay: float = 0.0):

    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)
        return SGDState(mu = mu, count = jnp.zeros([]))

    def update_fn(grads, state, params = None, learning_rate = learning_rate, momentum = momentum, weight_decay = weight_decay):

        # get the running gradients and count
        mu, count = state.mu, state.count + 1

        # normalize the gradients with layerwise learning rates
        grads_scaled = jax.tree.map(
            lambda lr, g: lr * g,
            lr_pytree, grads
        )

        # update momentum
        mu_next = jax.tree.map(
            lambda m, g: momentum * m + g,
            mu, grads_scaled
        )

        # multiply with the global learning rate
        updates = jax.tree.map(
            lambda m: -learning_rate * m, 
            mu_next
        )

        # Apply independent weight decay
        if params is not None:
            updates = jax.tree.map(
                lambda u, p: u - learning_rate * weight_decay * p,
                updates, params
            )

        return updates, SGDState(mu = mu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)

