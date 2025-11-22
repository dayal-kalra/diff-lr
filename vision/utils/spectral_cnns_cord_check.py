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
    return sum(x.size for x in jax.tree_leaves(params))

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
        kernel_init = nn.initializers.normal(stddev = jnp.sqrt(scaled_varw)) 
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
        kernel_init = nn.initializers.normal(stddev = jnp.sqrt(scaled_varw)) 
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


class ConvBlock(nn.Module):
  """ Convolution block: Convolution followed by activation """
  Conv: ModuleDef
  features: int
  act: Callable = nn.relu
  kernel_size: Tuple[int, int] = (3, 3)
  strides: Tuple[int, int] = (1, 1) # strides
  varw: float = 2.0
  use_bias: bool = True
  dtype: Type = jnp.float32

  @nn.compact
  def __call__(self, x,):
    x = self.Conv(
            features = self.features, 
            kernel_size = self.kernel_size, 
            strides = self.strides, 
            varw = self.varw, 
            use_bias = self.use_bias,
            dtype = self.dtype
            )(x)
    x = self.act(x)
    return x

class Myrtle(nn.Module):
    """ Myrtle CNNs in Standard Parameterization """
    width: int # number of filters
    num_layers: int # number of layers
    num_classes: int # number of classes
    pool_list: Sequence[int] # pooling list
    Conv: ModuleDef = SpectralConv
    kernel_size: Tuple[int, int] = (3, 3) # kernel size
    use_bias: bool = False # wheather to use bias or not
    act: Callable = nn.relu # activation
    varw: float = 2.0 # variance of the weights
    dtype: Type = jnp.float32

    @nn.compact
    def __call__(self, x):        
        ## forward pass
        for i in range(self.num_layers):
            x = ConvBlock(
                Conv = self.Conv, 
                features = self.width, 
                act = self.act, 
                kernel_size = self.kernel_size, 
                varw = self.varw, 
                use_bias = self.use_bias
                )(x)
            if i in self.pool_list:
                x = nn.avg_pool(x, (2 ,2) , (2 ,2))
            
            self.sow('intermediates', f'conv{i}', x)
        # use mean to apply average pooling
        x = jnp.mean(x, axis = (1, 2)) 
        # last layer
        x = SpectralDense(
            features = self.num_classes, 
            use_bias = self.use_bias,
            varw = 1.0,
            dtype = self.dtype,
            )(x)
        self.sow('intermediates', f'head', x)
        return jnp.asarray(x)
    
Myrtle5 = partial(Myrtle, pool_list = [1, 2, 3], num_layers = 4)
Myrtle7 = partial(Myrtle, pool_list = [1, 3, 5], num_layers = 6)
Myrtle10 = partial(Myrtle, pool_list = [2, 5, 8], num_layers = 9)