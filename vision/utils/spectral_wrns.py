from flax import linen as nn
import jax.numpy as jnp
from jax.tree_util import tree_map
import jax
from typing import Any, Callable, Sequence, Tuple, Type
from functools import partial

ModuleDef = Any

activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x, 'gelu': partial(nn.gelu, approximate = True), 'leaky_relu': nn.leaky_relu}
pools = {'max_pool': nn.max_pool, 'avg_pool': nn.avg_pool}
    
def count_parameters(params):
    "counts the number of parameters of a model"
    return sum(x.size for x in jax.tree.leaves(params))

class SpectralDense(nn.Module):
    """ 
    Dense layer in spectral parameterization
    The initial variance is to be scaled as var ~ 1/fan_in * min(1, fan_out, fan_in) and learning rate as fan_out/fan_in

    Remark: requires learning rate to be scaled as O(fan_out / fan_in) 
    NOTE: Truncated normal will introduce a factor of 1/1.3 in the variance; remove if required
    """
    features: int # fan_ou
    use_bias: bool = False
    varw: float = 1.0
    dtype: Type = jnp.float32 # bfloat16 can also be used; but inverse is not supported
    
    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        fan_out = self.features
        scaled_varw = self.varw * (1 / fan_in) * min(1, fan_out / fan_in)
        # print(x.shape, fan_in, fan_out, 1/scaled_varw)
        kernel_init = nn.initializers.truncated_normal(stddev = jnp.sqrt(scaled_varw)) 
        x = nn.Dense(
            features = self.features, 
            use_bias = self.use_bias, 
            kernel_init = kernel_init,
            dtype = self.dtype, 
            param_dtype = self.dtype,
            )(x)
        return x

class SpectralConv(nn.Module):
    """ Conv layer in spectral parameterization
    The initial variance is to be scaled as var ~ 1/fan_in * min(1, fan_out, fan_in) and learning rate as fan_out/fan_in

    Remark: requires learning rate to be scaled as O(fan_out / fan_in)  
    NOTE: Truncated normal will introduce a factor of 1/1.3 in the variance; remove if required
    """
    features: int # number of filters
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1) # strides
    varw: float = 1.0 # variance
    use_bias: bool = False
    dtype: Type = jnp.float32

    @nn.compact
    def __call__(self, x):

        fan_in = x.shape[-1] * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.features * self.kernel_size[0] * self.kernel_size[1]
        scaled_varw = self.varw * (1 / fan_in) * min(1, fan_out / fan_in)
        # print(x.shape, fan_in, fan_out, 1/scaled_varw)
        kernel_init = nn.initializers.truncated_normal(stddev = jnp.sqrt(scaled_varw)) 
        x = nn.Conv(
            features = self.features, 
            kernel_size = self.kernel_size, 
            strides = self.strides, 
            padding = 'SAME', 
            use_bias = self.use_bias,
            kernel_init = kernel_init, 
            dtype = self.dtype, 
            param_dtype = self.dtype,
            )(x)
        return x


class WideResNetBlock(nn.Module):
    """ WideResNet Block in Spectral parameterization """
    features: int # number of filters
    Conv: ModuleDef = SpectralConv
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1) # strides
    use_bias: bool = True
    act: Callable = nn.relu # activation
    varw: float = 2.0 
    dtype: Type = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        # First block: Conv + LN + act

        y = self.Conv(
            features = self.features, 
            kernel_size = self.kernel_size, 
            strides = self.strides, 
            varw = self.varw, 
            use_bias = self.use_bias,
            dtype = self.dtype
            )(x)

        y = nn.LayerNorm()(y)
        y = self.act(y)

        # Second block: Conv + LN + act

        y = self.Conv(
            features = self.features, 
            kernel_size = self.kernel_size, 
            # strides not used; default (1, 1)
            varw = self.varw, 
            use_bias = self.use_bias,
            dtype = self.dtype
            )(y)

        y = nn.LayerNorm()(y)

        # reshape the inputs x to have the same dimension as the output y
        if x.shape[-1] != self.features or self.strides != (1, 1):
            x = self.Conv(
                features = self.features, 
                kernel_size = (1, 1), 
                strides = self.strides, 
                varw = 1.0, 
                use_bias = self.use_bias,
                dtype = self.dtype
                )(x)
        # residual connection and then apply activation
        return self.act(y + x)


class WideResNet(nn.Module):
    """ WideResNet in SP """
    stage_sizes: Sequence[int]
    num_filters: int
    widening_factor: int
    num_classes: int
    Conv: ModuleDef = SpectralConv
    kernel_size: Tuple[int, int] = (3, 3)
    use_bias: bool = False
    act: Callable = nn.relu # activation
    varw: float = 2.0
    dtype: Type = jnp.float32
    
    @nn.compact
    def __call__(self, x):

        # First Conv + LN + act
        x = self.Conv(
            features = self.num_filters, 
            kernel_size = self.kernel_size,
            varw = self.varw, 
            use_bias = self.use_bias,
            dtype = self.dtype,
            )(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)

        # Stages
        for stage, num_blocks in enumerate(self.stage_sizes):
            # Blocks
            for block in range(num_blocks):
                # width and strides in different stages
                features = self.num_filters * (2 ** stage) * self.widening_factor
                strides = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = WideResNetBlock(
                    features = features, 
                    Conv = self.Conv, 
                    kernel_size = self.kernel_size, 
                    strides = strides, 
                    use_bias = self.use_bias, 
                    act = self.act, 
                    varw = self.varw,
                    dtype = self.dtype,
                    )(x)
                
        # take a global average along axes [1, 2]
        x = jnp.mean(x, axis = (1, 2))
        x = SpectralDense(
            features = self.num_classes, 
            use_bias = self.use_bias,
            varw = 1.0,
            dtype = self.dtype,
            )(x) 
        return jnp.asarray(x)

WideResNet12 = partial(WideResNet, stage_sizes = [1, 1, 1])
WideResNet16 = partial(WideResNet, stage_sizes = [2, 2, 2])
WideResNet20 = partial(WideResNet, stage_sizes = [3, 3, 3])
WideResNet28 = partial(WideResNet, stage_sizes = [4, 4, 4])
WideResNet40 = partial(WideResNet, stage_sizes = [6, 6, 6])