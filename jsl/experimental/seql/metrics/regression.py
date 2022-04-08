import jax.numpy as jnp
from jax import random, vmap

import chex
from typing import Tuple

import dataclasses

from jsl.experimental.seql.metrics.base import MetricFn
from jsl.experimental.seql.metrics.utils import average_sampled_log_likelihood
from jsl.experimental.seql.utils import gaussian_log_likelihood


@dataclasses.dataclass
class GaussianSampleKL(MetricFn):
    """Evaluates KL according to optimized Gaussian residual model."""

    def __call__(self,
                 predictions: chex.Array,
                 test_data: Tuple[Tuple[chex.Array, chex.Array], float]) -> float:
        """Computes KL estimate on a single instance of test data."""
        (x, y), true_ll = test_data
        tau = x.shape[0]
        batched_err = predictions - jnp.expand_dims(y, 0)
        chex.assert_shape(batched_err, [self.num_enn_samples, tau, 1])

        # ENN uses the enn_sigma to compute likelihood of sampled data
        enn_mean = jnp.zeros((tau,))
        enn_cov = self.enn_sigma ** 2 * jnp.eye(tau)
        batched_ll = vmap(gaussian_log_likelihood, in_axes=[None, None, 0])

        sampled_ll = batched_ll(enn_mean, enn_cov, batched_err)
        chex.assert_shape(sampled_ll, [self.num_enn_samples, 1])

        ave_ll = average_sampled_log_likelihood(sampled_ll)
        return true_ll - ave_ll
