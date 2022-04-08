import jax.numpy as jnp
from jax import random

import chex
from typing import Tuple, Optional

from jsl.experimental.seql.environments.sequential_data_env import SequentialDataEnvironment
from jsl.experimental.seql.utils import gaussian_log_likelihood


class SequentialRegressionEnvironment(SequentialDataEnvironment):
    def __init__(self,
                 X_train: chex.Array,
                 y_train: chex.Array,
                 X_test: chex.Array,
                 y_test: chex.Array,
                 true_model: chex.Array,
                 train_batch_size: int,
                 test_batch_size: int,
                 obs_noise: float,
                 key: Optional[chex.PRNGKey] = None,
                 tau: int = 1):

        super().__init__(X_train,
                         y_train,
                         X_test,
                         y_test,
                         true_model,
                         train_batch_size,
                         test_batch_size,
                         key,
                         tau)
        self.obs_noise = obs_noise

    def reward(self,
               predictions: chex.Array,
               t: int,
               train: bool = False):

        cov = self.obs_noise * jnp.eye(len(predictions))
        if train:
            mu = self.y_train[t]  # - predictions
        else:
            mu = self.y_test[t]  # - predictions
        return gaussian_log_likelihood(mu, cov, predictions)

    def test_data(self, key: chex.PRNGKey) -> Tuple[Tuple[chex.Array, chex.Array], float]:
        # https://github.com/deepmind/neural_testbed/blob/7defebf0a232a921d720b870c3ab3a56d38e7ceb/neural_testbed/generative/gp_regression_envlikelihood.py#L86
        """Generates test data and evaluates log likelihood w.r.t. environment.
        The test data that is output will be of length tau examples.
        We wanted to "pass" tau here... but ran into jax.jit issues.
        Args:
          key: Random number generator key.
        Returns:
          Tuple of data (with tau examples) and log-likelihood under posterior.
        """

        x_key, y_key = random.split(key, 2)

        out = self.y_test.shape[-1]
        nfeatures = self.X_test.shape[-1]

        X_test = self.X_test.reshape((-1, nfeatures))
        y_test = self.y_test.reshape((-1, out))

        # Sample tau x's from the testing cache for evaluation
        test_x_indices = random.randint(x_key, [self.tau], 0, len(X_test))
        X_test = X_test[test_x_indices]
        chex.assert_shape(X_test, [self.tau, nfeatures])

        # Sample y_function for the test data
        y_test = y_test[test_x_indices]
        chex.assert_shape(y_test, [self.tau, 1])

        # Compute the log likelihood with respect to the environment
        cov = self.obs_noise * jnp.eye(self.tau)
        chex.assert_shape(cov, [self.tau, self.tau])

        log_likelihood = gaussian_log_likelihood(self.true_model(X_test), cov, y_test)

        return (X_test, y_test), log_likelihood