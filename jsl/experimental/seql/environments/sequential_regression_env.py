import jax.numpy as jnp
from jax import random, vmap

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
                 key: Optional[chex.PRNGKey] = None):

        super().__init__(X_train,
                         y_train,
                         X_test,
                         y_test,
                         true_model,
                         train_batch_size,
                         test_batch_size,
                         key)
        self.obs_noise = obs_noise


    def get_joint_data(self,
                       key: chex.PRNGKey,
                       nsamples: int,
                       njoint: int
                       ) -> Tuple[Tuple[chex.Array, chex.Array], float]:
        # https://github.com/deepmind/neural_testbed/blob/7defebf0a232a921d720b870c3ab3a56d38e7ceb/neural_testbed/generative/gp_regression_envlikelihood.py#L86
        """Generates test data and evaluates log likelihood w.r.t. environment.
        The test data that is output will be of length njoint examples.
        We wanted to "pass" njoint here... but ran into jax.jit issues.
        Args:
          key: Random number generator key.
        Returns:
          Tuple of data (with njoint examples) and log-likelihood under posterior.
        """
        def joint_data_per_sample(key):
            x_key, y_key = random.split(key, 2)

            out = self.y_test.shape[-1]
            nfeatures = self.X_test.shape[-1]

            X_test = self.X_test.reshape((-1, nfeatures))
            y_test = self.y_test.reshape((-1, out))

            # Sample njoint x's from the testing cache for evaluation
            test_x_indices = random.randint(x_key, [njoint], 0, len(X_test))
            X_test = X_test[test_x_indices]
            chex.assert_shape(X_test, [njoint, nfeatures])

            # Sample y_function for the test data
            y_test = y_test[test_x_indices]
            chex.assert_shape(y_test, [njoint, 1])

            # Compute the log likelihood with respect to the environment
            cov = self.obs_noise * jnp.eye(njoint)
            chex.assert_shape(cov, [njoint, njoint])

            log_likelihood = gaussian_log_likelihood(self.true_model(X_test), cov, y_test)

            return (X_test, y_test), log_likelihood

        keys = random.split(key, nsamples)

        return vmap(joint_data_per_sample)(keys)
