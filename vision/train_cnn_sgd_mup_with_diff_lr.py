# in use imports
import utils.mup_cnns as model_utils
import utils.train_utils as train_utils
import utils.data_utils as data_utils
import utils.loss_utils as loss_utils
import utils.schedules_utils as schedules_utils
import utils.optim_utils as optim_utils
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

from utils.diff_lr import LRPredictor, compute_features
from pathlib import Path
import orbax.checkpoint

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
import math

# for deterministic gpu computations
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

"""Model definition and train state definition"""

# After initializing your model
def print_params_info(params, prefix = ""):
    """Recursively print parameter paths, shapes, and variances."""
    if isinstance(params, dict):
        for key, value in params.items():
            new_prefix = f"{prefix}/{key}" if prefix else f"{key}"
            print_params_info(value, new_prefix)
    elif hasattr(params, "shape"): 
        if len(params.shape) == 4:
            h, w, fan_in, fan_out = params.shape
            scaled_fan_in = h * w * fan_in
            scaled_fan_out = h * w * fan_out 
        elif len(params.shape) == 2:
            scaled_fan_in, scaled_fan_out = params.shape
        elif len(params.shape) == 1:
            scaled_fan_in = params.shape[0]
            scaled_fan_out = params.shape[0]
        else:
            print(f'Unknown')

        print(f"Path: {prefix}")
        print(f"  Shape: {params.shape}")
        print(f"  1/Variance: {1/jnp.var(params):.8f}")
        print(f" Scaled fan_in: {scaled_fan_in}")
        print(f" Scaled fan_out: {scaled_fan_out}")
        print()

def get_lr_scale(param):
    return 1.0

def load_lr_predictor(lr_predictor_ckpt_path, gru_hidden_size=32, mlp_hidden_size=64, lr_min=1e-5, lr_max=1.0):
    """Load LR predictor parameters from checkpoint"""
    lr_predictor = LRPredictor(
        gru_hidden_size=gru_hidden_size,
        mlp_hidden_size=mlp_hidden_size,
        lr_min=lr_min,
        lr_max=lr_max
    )
    
    dummy_features = {
        'weight_norm': 1.0, 'grad_norm': 1.0, 'loss_current': 1.0,
        'step_progress': 0.5, 'loss_history': jnp.ones(100)
    }
    
    key = jax.random.PRNGKey(42)
    lr_predictor_params = lr_predictor.init(key, dummy_features)['params']
    
    checkpoint_path = Path(lr_predictor_ckpt_path).resolve()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored_full = checkpointer.restore(checkpoint_path)
    lr_predictor_params = restored_full['params']
    
    print(f"Loaded LR predictor from {lr_predictor_ckpt_path}")
    return lr_predictor, lr_predictor_params

def create_train_state(cfg: argparse.ArgumentParser, batch: Tuple):
    x, y = batch

    x = x[:cfg.batch_size, ...]
    y = y[:cfg.batch_size, ...]
    
    # initialize a model
    model = models[cfg.model](width = cfg.width, num_classes = cfg.out_dim, act = cfg.act, varw = cfg.varw)
    # initialize using the init seed
    key = jax.random.PRNGKey(cfg.init_seed)
    init_params = model.init(key, x)['params']
    # print_params_info(init_params)
    

    # count the number of parameters
    num_params = model_utils.count_parameters(init_params)
    print(f'The model has {num_params/1e6:0.4f}M parameters')

    shapes = jax.tree.map(lambda x: x.shape, init_params)
    # print(shapes)

    # create an optimizer
    lr_pytree = jax.tree.map(get_lr_scale, init_params)
    # print(lr_pytree)

    sgd = partial(optim_utils.SGD, lr_pytree = lr_pytree)
    
    base_optim = optax.inject_hyperparams(sgd)(learning_rate = cfg.lr_init, momentum = cfg.momentum, weight_decay = cfg.weight_decay)

    optim = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        base_optim,
    )
    # create a train state
    state = train_utils.TrainState.create(apply_fn = model.apply, params = init_params, opt = optim)

    return state, num_params


def train_and_evaluate(cfg: argparse.ArgumentParser, train_ds: Tuple, test_ds: Tuple, lr_predictor, lr_predictor_params):
    "train model acording the cfg"
    
    # create a train state
    state, num_params = create_train_state(cfg, train_ds)
    
    # create train and test batches for measurements: measure batches are called train_batches and val_batches; training batches are called batches
    seed = cfg.sgd_seed

    x_train, y_train = train_ds
    train_eval_ds = x_train[:cfg.num_test], y_train[:cfg.num_test]
    train_loader = train_utils.data_stream(seed, train_ds, cfg.batch_size, augment = cfg.use_augment)
    train_eval_loader = train_utils.data_stream(seed, train_eval_ds, cfg.batch_size, augment = False)
    test_loader = train_utils.data_stream(seed, test_ds, cfg.batch_size, augment = False)

    ########### TRAINING ##############

    # save a checkpoint at initialization
    if cfg.save_ckpt:
        print(f'Saving initial checkpoint to {cfg.ckpt_dir}')
        save_checkpoint(train_state = state, step = 0, checkpoint_dir = cfg.ckpt_dir)

    # store training results
    train_results = list()
    eval_results = list()

    divergence = False
    
    running_loss = 0.0

    # Initialize loss history for LR predictor
    loss_history = jnp.zeros(100)
    lr_step = None

    for step in range(cfg.num_steps):  

        epoch = (step // cfg.num_batches) + 1 
        cosine_step = state.step - cfg.warmup_steps + 1

        batch = next(train_loader)
        imgs, targets = batch

        # Use LR predictor
        grads, loss_step = train_utils.grads_step(state, batch, cfg.loss_fn)

        features = compute_features(state.params, grads, loss_step, loss_history, step, cfg.num_steps)
        if lr_step is None or step % 10 == 0:
            lr_step = jnp.squeeze(lr_predictor.apply({'params': lr_predictor_params}, features))

        # update the learning rate
        state.opt_state[-1].hyperparams['learning_rate'] = lr_step
        state = state.apply_gradients(grads=grads)

        # estimate weight norm after the step
        # flatten the state params and then compute the norm
        weight_norm = jnp.sqrt(sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(state.params)]))

        # print loss and learning rate every step
        # print(f'step: {state.step}, lr_step: {lr_step:0.6f}, loss_step: {loss_step:0.4f}, weight_norm: {weight_norm:0.4f}')

        # Update loss history
        loss_history = jnp.roll(loss_history, -1)
        loss_history = loss_history.at[-1].set(loss_step)

        # store training results
        result = np.array([state.step, epoch, lr_step, loss_step])
        train_results.append(result)
        
        #check for divergence
        if (jnp.isnan(loss_step) or jnp.isinf(loss_step)): divergence = True; break

        running_loss += loss_step

        if state.step % cfg.num_batches == 0 or step == 0 or (step+1 == cfg.num_steps):

            # estiamte the running loss and reset the parameters
            train_loss, train_accuracy = train_utils.compute_eval_metrics_dataset(state, train_eval_loader, cfg.loss_fn, cfg.num_test, cfg.batch_size)

            # estimate test accuracy
            test_loss, test_accuracy = train_utils.compute_eval_metrics_dataset(state, test_loader, cfg.loss_fn, cfg.num_test, cfg.batch_size)
            print(f't: {state.step}, lr_step: {lr_step:0.4f}, training loss: {train_loss:0.4f}, test_loss: {test_loss:0.4f}, test_accuracy: {test_accuracy:0.4f}')
            result = np.asarray([state.step, epoch, lr_step, train_loss, test_loss, test_accuracy])
            eval_results.append(result)

        # save checkpoint
        if state.step in cfg.ckpt_steps or (step+1) == cfg.num_steps and cfg.save_ckpt:
            save_checkpoint(train_state = state, step = state.step, checkpoint_dir = cfg.ckpt_dir)
    
    train_results = np.asarray(train_results)
    eval_results = np.asarray(eval_results)
    
    return divergence, train_results, eval_results, num_params

models = {'Myrtle5': model_utils.Myrtle5, 'Myrtle7': model_utils.Myrtle7, 'Myrtle10': model_utils.Myrtle10}
loss_fns = {'mse': loss_utils.mse_loss, 'xent': loss_utils.cross_entropy_loss}
activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x}
decay_schedules = {'polynomial': schedules_utils.polynomial_decay, 'cosine': schedules_utils.cosine_decay}

parser = argparse.ArgumentParser(description = 'Experiment parameters')
parser.add_argument('--cluster', type = str, default = 'zaratan')
# Dataset parameters
parser.add_argument('--dataset_name', type = str, default = 'cifar-10')
parser.add_argument('--out_dim', type = int, default = 10)
parser.add_argument('--augment', type = str, default = 'False')
# Model parameters
parser.add_argument('--abc', type = str, default = 'mup')
parser.add_argument('--width', type = int, default = 16)
parser.add_argument('--depth', type = int, default = 5)
parser.add_argument('--bias', type = str, default = 'False') # careful about the usage
parser.add_argument('--act_name', type = str, default = 'relu')
parser.add_argument('--init_seed', type = int, default = 42)
parser.add_argument('--varw', type = float, default = 2.0)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'xent')
parser.add_argument('--opt_name', type = str, default = 'sgd')
parser.add_argument('--sgd_seed', type = int, default = 1337)
parser.add_argument('--warmup_steps', type = int, default = 4000)
parser.add_argument('--warmup_exponent', type = float, default = 1.0) # exponent for warmup
parser.add_argument('--decay_schedule_name', type = str, default = 'cosine') # decay schedule
parser.add_argument('--decay_exponent', type = float, default = 1.0) # exponent for decay
parser.add_argument('--num_steps', type = int, default = 100_000)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_peak', type = float, default = 0.01)
parser.add_argument('--lr_min_factor', type = lambda x: float('inf') if x.lower() == 'inf' else float(x), default = float('inf'))
parser.add_argument('--grad_clip', type=float, default = 1.0)
parser.add_argument('--momentum', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--weight_decay', type=float, default = 0.0)
# Evaluation
parser.add_argument('--eval_interval', type = int, default = 1000)
# Sharpness estimation
parser.add_argument('--rerun', type = str, default = 'False')
parser.add_argument('--save_dir', type=str, default='cnn_results/learned_lr')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints/learned_lr')
parser.add_argument('--save_ckpt', type = lambda x: False if x.lower() == 'false' else True)
parser.add_argument('--measure_batches', type = int, default = 10)

parser.add_argument('--lr_predictor_ckpt_path', type=str, required=True)
parser.add_argument('--gru_hidden_size', type=int, default=32)
parser.add_argument('--mlp_hidden_size', type=int, default=64)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--lr_max', type=float, default=10.0)

cfg = parser.parse_args()

# Model parameters
cfg.model = f'Myrtle{cfg.depth}'
cfg.use_bias = True if cfg.bias == 'True' else False
cfg.use_augment = True if cfg.augment == 'True' else False
cfg.act = model_utils.activations[cfg.act_name]
# define loss
cfg.loss_fn = loss_fns[cfg.loss_name]
# define decay schedule
cfg.decay_schedule = decay_schedules[cfg.decay_schedule_name]
# set mining learning rate
cfg.lr_min = cfg.lr_peak / cfg.lr_min_factor

# create save directories if they do not exist

os.makedirs(cfg.save_dir, exist_ok=True)

# Dataset loading 
cfg.ds_dir = 'data/'

(x_train, y_train), (x_test, y_test) = data_utils.load_image_data(cfg.ds_dir, cfg.dataset_name, flatten = False, subset = False)

cfg.num_train, cfg.num_test = x_train.shape[0], x_test.shape[0]

# standardize the inputs
x_train = data_utils._standardize(x_train, abc = 'sp')
x_test = data_utils._standardize(x_test, abc = 'sp')

cfg.in_dim = jnp.array(x_train.shape[1:])
cfg.out_dim = len(jnp.unique(y_train))
cfg.num_train = x_train.shape[0]
cfg.num_test = x_test.shape[0]

# one hot encoding for the labels
y_train = jax.nn.one_hot(y_train, cfg.out_dim)
y_test = jax.nn.one_hot(y_test, cfg.out_dim)

cfg.num_batches = train_utils.estimate_num_batches(cfg.num_train, cfg.batch_size)

print(cfg)

trained_for = cfg.lr_predictor_ckpt_path.split('step_')[1]
### TRAIN THE NETWORK AND EVALUATE ####

cfg.train_path = f'{cfg.save_dir}/train_step{trained_for}_{cfg.dataset_name}_{cfg.model}_{cfg.abc}_n{cfg.width}_d{cfg.depth}_bias{cfg.use_bias}_{cfg.act_name}_I{cfg.init_seed}_J{cfg.sgd_seed}_{cfg.loss_name}_aug{cfg.augment}_{cfg.opt_name}_lr{cfg.lr_peak:0.1e}_lr{cfg.lr_min_factor}_k{cfg.warmup_exponent}_{cfg.decay_schedule_name}_p{cfg.decay_exponent}_Tw{cfg.warmup_steps}_T{cfg.num_steps}_B{cfg.batch_size}_m{cfg.momentum}_gc{cfg.grad_clip}_wd{cfg.weight_decay}.tab'    
cfg.eval_path = f'{cfg.save_dir}/eval_step{trained_for}_{cfg.dataset_name}_{cfg.model}_{cfg.abc}_n{cfg.width}_d{cfg.depth}_bias{cfg.use_bias}_{cfg.act_name}_I{cfg.init_seed}_J{cfg.sgd_seed}_{cfg.loss_name}_aug{cfg.augment}_{cfg.opt_name}_lr{cfg.lr_peak:0.1e}_lr{cfg.lr_min_factor}_k{cfg.warmup_exponent}_{cfg.decay_schedule_name}_p{cfg.decay_exponent}_Tw{cfg.warmup_steps}_T{cfg.num_steps}_B{cfg.batch_size}_m{cfg.momentum}_gc{cfg.grad_clip}_wd{cfg.weight_decay}.tab'    
# checkpoint filename
cfg.ckpt_dir = f'{cfg.ckpt_dir}/step{trained_for}_{cfg.dataset_name}_{cfg.model}_{cfg.abc}_n{cfg.width}_d{cfg.depth}_bias{cfg.use_bias}_{cfg.act_name}_I{cfg.init_seed}_J{cfg.sgd_seed}_{cfg.loss_name}_aug{cfg.augment}_{cfg.opt_name}_lr{cfg.lr_peak:0.1e}_lr{cfg.lr_min_factor}_k{cfg.warmup_exponent}_{cfg.decay_schedule_name}_p{cfg.decay_exponent}_Tw{cfg.warmup_steps}_T{cfg.num_steps}_B{cfg.batch_size}_m{cfg.momentum}_gc{cfg.grad_clip}_wd{cfg.weight_decay}'

# when to save checkpoints
# for the first 10000 steps, save every 1000 steps
# after that, save every 10000 steps
cfg.ckpt_steps = [i for i in range(0, 10_000, 1_000)] + [i for i in range(10_000, cfg.num_steps+1, 10_000)]

# Load LR predictor
lr_predictor, lr_predictor_params = load_lr_predictor(
    cfg.lr_predictor_ckpt_path, cfg.gru_hidden_size, cfg.mlp_hidden_size, cfg.lr_min, cfg.lr_max
)

divergence, train_results, eval_results, num_params = train_and_evaluate(
    cfg, (x_train, y_train), (x_test, y_test), lr_predictor, lr_predictor_params
)
# save training data
df_train = pd.DataFrame(train_results, columns = ['step', 'epoch', 'lr', 'loss_step'])
df_train['num_params'] = num_params
df_train['depth'] = cfg.depth
df_train['width'] = cfg.width
df_train.to_csv(cfg.train_path, sep = '\t')
        
# save eval data
df_eval = pd.DataFrame(eval_results, columns = ['step', 'epoch', 'lr', 'train_loss', 'test_loss', 'test_accuracy'])
df_eval['num_params'] = num_params
df_eval.to_csv(cfg.eval_path, sep = '\t')
