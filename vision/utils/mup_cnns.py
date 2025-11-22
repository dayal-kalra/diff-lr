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

def rms_norm(x, axes):
    return jnp.sqrt(jnp.mean(x**2, axis=axes)).mean()

class muDense(nn.Module):
    """ muP Readout layer """
    fan_out: int
    eff_width: int  # mup width
    use_bias: bool = True  # bool for bias
    varw: float = 1.0  # variance
    dtype: Type = jnp.float32

    def setup(self):
        # 1 / fan_in init
        kernel_init = jax.nn.initializers.normal(
            stddev=jnp.sqrt(self.varw / self.eff_width)
        )
        self.dense = nn.Dense(
            self.fan_out,
            kernel_init=kernel_init,
            use_bias=self.use_bias,
            dtype=self.dtype
        )

    def __call__(self, x):
        return self.dense(x)

class muConv(nn.Module):
    """ muP Conv layer with 1 / width initialization """
    fan_out: int  # number of filters
    eff_width: int  # the actual width; estimated before hand
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)  # strides
    varw: float = 1.0  # variance
    use_bias: bool = True

    def setup(self):
        # varw / width initialization
        kernel_init = jax.nn.initializers.normal(
            stddev=jnp.sqrt(self.varw / self.eff_width)
        )
        self.conv = nn.Conv(
            self.fan_out,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME',
            kernel_init=kernel_init,
            use_bias=self.use_bias
        )

    def __call__(self, x):
        return self.conv(x)  

###########################################
############## Myrtle CNNs ################
###########################################

class ConvBlock(nn.Module):
    """ Convolution block: Convolution followed by activation """
    Conv: ModuleDef
    width: int
    eff_width: int  # Added: specify eff_width beforehand
    act: Callable = nn.relu
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)  # strides
    varw: float = 2.0
    use_bias: bool = True

    def setup(self):
        self.conv = self.Conv(
            fan_out=self.width,
            eff_width=self.eff_width,
            kernel_size=self.kernel_size,
            strides=self.strides,
            varw=self.varw,
            use_bias=self.use_bias
        )

    def __call__(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class Myrtle(nn.Module):
    """ Myrtle CNNs in mu Parameterization """
    width: int  # number of filters
    num_layers: int  # number of layers
    num_classes: int  # number of classes
    pool_list: Sequence[int]  # pooling list
    kernel_size: Tuple[int, int] = (3, 3)  # kernel size
    use_bias: bool = False  # whether to use bias or not
    act: Callable = nn.relu  # activation
    varw: float = 2.0  # variance of the weights

    def setup(self):
        self.eff_width = self.width * self.kernel_size[0] * self.kernel_size[1]
        
        # Create conv blocks
        self.conv_blocks = [
            ConvBlock(
                width=self.width,
                Conv=muConv,
                eff_width=self.eff_width,
                act=self.act,
                kernel_size=self.kernel_size,
                use_bias=self.use_bias,
                varw=self.varw,
            )
            for _ in range(self.num_layers)
        ]
        
        # Create dense layer
        self.dense = muDense(
            fan_out=self.num_classes,
            eff_width=self.eff_width,
            varw=1.0,
            use_bias=self.use_bias,
        )

    def __call__(self, x, capture_activations: bool = False):
        activations = {} if capture_activations else None
        # add inputs to activations
        if capture_activations:
            activations['x'] = x

        # mup sqrt scaling for inputs
        x /= jnp.sqrt(x.shape[-1] * self.kernel_size[0] * self.kernel_size[1])
        
        ## forward pass
        for i in range(self.num_layers):
            x = self.conv_blocks[i](x)
            if i == 0:
                # mup sqrt scaling for the first layer
                x *= jnp.sqrt(self.eff_width)

            # pooling
            if i in self.pool_list:
                x = nn.avg_pool(x, (2, 2), (2, 2))
            
            # save activations
            if capture_activations:
                activations[f'h{i+1}'] = x
                
        # use mean to apply average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # last layer
        logits = self.dense(x)
        
        # mup sqrt scaling
        logits /= jnp.sqrt(self.eff_width)
        if capture_activations:
            return logits, activations
        return jnp.asarray(logits)

Myrtle5 = partial(Myrtle, pool_list=[1, 2, 3], num_layers=4)
Myrtle7 = partial(Myrtle, pool_list=[1, 3, 5], num_layers=6)
Myrtle10 = partial(Myrtle, pool_list=[2, 5, 8], num_layers=9)
