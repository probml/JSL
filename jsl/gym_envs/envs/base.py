# It is based on
# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/generative/classification_envlikelihood.py
# Includes classification-based and regression-based testbed based around a logit_fn, fit_fn and x_generator.

import jax
from jax import vmap, nn
from jax import random
import jax.numpy as jnp

import haiku as hk

import chex

from typing import Union, List, Callable

import chex
from sklearn.covariance import log_likelihood


def mean_squared_error(pred: chex.Array, target: chex.Array):
    err = pred - target
    return jnp.mean(jnp.square(err))  # mse


def categorical_log_likelihood(probs: chex.Array, labels: chex.Array):
    """Computes joint log likelihood based on probs and labels."""
    num_data, unused_num_classes = probs.shape
    assert len(labels) == num_data
    assigned_probs = probs[jnp.arange(num_data), jnp.squeeze(labels)]
    return jnp.sum(jnp.log(assigned_probs))

def sample_gaussian_cls_data(apply_fn: Callable,
                         x_generator: Callable,
                         num_train: int,
                         key: chex.PRNGKey):
    """Generates training data for given problem."""
    x_key, y_key = jax.random.split(key, 2)

    # Checking the dimensionality of our data coming in.
    x_train = x_generator(x_key, num_train)
    input_dim = x_train.shape[1]
    chex.assert_shape(x_train, [num_train, input_dim])

    # Generate environment function across x_train
    train_logits = apply_fn(x_train)  # [n_train, n_class]
    num_classes = train_logits.shape[-1]  # Obtain from logit_fn.
    chex.assert_shape(train_logits, [num_train, num_classes])
    train_probs = jax.nn.softmax(train_logits)

    # Generate training data.
    def sample_output(probs: chex.Array, key: chex.PRNGKey) -> chex.Array:
      return jax.random.choice(key, num_classes, shape=(1,), p=probs)
    y_keys = jax.random.split(y_key, num_train)
    y_train = jax.vmap(sample_output)(train_probs, y_keys)

    # Compute the log likelihood with respect to the environment
    log_likelihood = categorical_log_likelihood(train_probs, y_train)
    return (x_train, y_train), train_probs, log_likelihood

def sample_gaussian_reg_data(apply_fn: Callable,
                         x_generator: Callable,
                         num_train: int,
                         key: chex.PRNGKey):
                
  """Generates training data for given problem."""
  # Checking the dimensionality of our data coming in.
  x_train = x_generator(key, num_train)

  input_dim = x_train.shape[1]  
  chex.assert_shape(x_train, [num_train, input_dim])

  # Generate environment function across x_train
  y_train = apply_fn(x_train)  # [n_train, n_class]

  return (x_train, y_train), None, None

def make_gaussian_sampler(input_dim: int):
    def gaussian_generator(key: chex.PRNGKey, num_samples: int) -> chex.Array:
        return random.normal(key, [num_samples, input_dim])

    return gaussian_generator


def make_mlp_apply_fn(
    input_dim: int,
    output_dim: int,
    temperature: float,
    hidden_layer_sizes: Union[int, List[int], None],
    key: chex.PRNGKey):
  """Factory method to create a generative model MLP."""
  
  hidden_layer_sizes = [hidden_layer_sizes] if type(hidden_layer_sizes)==int else hidden_layer_sizes

  def linear_fn(x: chex.Array):
    """Defining the generative model Linear."""
    return hk.Linear(output_dim)(x)

  # Generating the logit function
  def net_fn(x: chex.Array):
    """Defining the generative model MLP."""
    
    hidden = hidden_layer_sizes[0]
    y = hk.Linear(
        output_size=hidden,
        b_init=hk.initializers.RandomNormal(1./jnp.sqrt(input_dim)),
    )(x)
    y = jax.nn.relu(y)

    for hidden in hidden_layer_sizes[1:]:
      y = hk.Linear(hidden_layer_sizes)(y)
      y = jax.nn.relu(y)
    return hk.Linear(output_dim)(y)
  
  if hidden_layer_sizes:
    transformed = hk.without_apply_rng(hk.transform(net_fn))
  else:
    transformed = hk.without_apply_rng(hk.transform(linear_fn))

  dummy_input = jnp.zeros([1, input_dim])
  params = transformed.init(key, dummy_input)

  def forward(x: chex.Array):
      return transformed.apply(params, x) / temperature

  apply_fn = jax.jit(forward)

  return apply_fn


def make_poly_fit_fn(
        degree: int,
        key: chex.PRNGKey,
        use_bias: bool = True
):
    w_key, b_key = random.split(key)
    ws = random.normal(w_key, (degree,))
    b = random.normal(b_key, (1,)) if use_bias else 0
    degrees = jnp.arange(degree)

    def fit_fn(x: chex.Array):
        def power_fn(w, x, k):
            return w * x ** k

        poly = vmap(power_fn, in_axes=(0, None, 0))(ws, x, degrees)
        return jnp.sum(poly, axis=-1) + b

    return fit_fn
