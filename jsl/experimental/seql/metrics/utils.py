import jax.numpy as jnp
from jax import random, vmap, lax

import chex


def average_sampled_log_likelihood(x: chex.Array) -> float:
    """Computes average log likelihood from samples.
    This method takes several samples of log-likelihood, converts
    them to likelihood (by exp), then takes the average, then
    returns the logarithm over the average  LogSumExp
    trick is used for numerical stability.
    Args:
      x: chex.Array
    Returns:
      log-mean-exponential
    """
    return lax.cond(
        jnp.isneginf(jnp.max(x)),
        lambda x: -jnp.inf,
        lambda x: jnp.log(jnp.mean(jnp.exp(x - jnp.max(x)))) + jnp.max(x),
        operand=x,
    )


def evaluate_quality(agent, env, belief, metric_fns, nsamples, ntest_seeds, key):
    def evaluate(key):
        (X, y), true_ll = env.test_data(key)
        samples = agent.sample_predict(key, belief, X, nsamples)

        metrics = {}
        for metric_fn in metric_fns:
            metric_name = metric_fn.name
            metrics[metric_name] = metric_fn(samples, (X, y), true_ll)
        return metrics

    keys = random.split(key, ntest_seeds)
    return jnp.mean(vmap(evaluate)(keys), axis=0)
