import jax.numpy as jnp
from jax import nn
from jax.scipy.stats import multivariate_normal

def classification_loss(logprobs, targets):
  nclasses = logprobs.shape[-1]
  one_hot_targets = nn.one_hot(targets, nclasses, axis=-1)
  nll = jnp.sum(logprobs * one_hot_targets, axis=-1)
  ce = -jnp.mean(nll)
  return ce

def regression_loss(predictions, mean, cov):
  return multivariate_normal.logpdf(predictions, mean, cov, allow_singular=None)
  # return jnp.mean(jnp.power(predictions - outputs, 2)) 