from typing import Optional
import jax.numpy as jnp
from jax import random, vmap

import chex
from typing import Tuple

from jsl.experimental.seql.environments.sequential_data_env import SequentialDataEnvironment
from jsl.experimental.seql.utils import categorical_log_likelihood


class SequentialClassificationEnvironment(SequentialDataEnvironment):

    def __init__(self,
                 X_train: chex.Array,
                 y_train: chex.Array,
                 X_test: chex.Array,
                 y_test: chex.Array,
                 true_model: chex.Array,
                 train_batch_size: int,
                 test_batch_size: int,
                 logprobs: chex.Array,
                 key: Optional[chex.PRNGKey] = None):
        ntrain = len(y_train)
        ntest = len(y_test)
        _, out = logprobs.shape

        ntrain_batches = ntrain // train_batch_size
        ntest_batches = ntest // test_batch_size

        self.train_logprobs = jnp.reshape(logprobs[:ntrain], [ntrain_batches, train_batch_size, out])
        self.test_logprobs = jnp.reshape(logprobs[ntrain:], [ntest_batches, test_batch_size, out])

        super().__init__(X_train,
                         y_train,
                         X_test,
                         y_test,
                         true_model,
                         train_batch_size,
                         test_batch_size,
                         key)

    def shuffle_data(self, key: chex.PRNGKey):
        super().shuffle_data(key)

        ntrain_batches, train_batch_size, out = self.y_train.shape
        ntest_batches, test_batch_size, _ = self.y_test.shape

        self.train_logprobs = jnp.reshape(self.train_logprobs, [-1, out])
        self.test_logprobs = jnp.reshape(self.test_logprobs, [-1, out])

        self.train_logprobs = self.train_logprobs[self.train_indices]
        self.test_logprobs = self.test_logprobs[self.test_indices]

        self.train_logprobs = jnp.reshape(self.train_logprobs, [ntrain_batches, train_batch_size, out])
        self.test_logprobs = jnp.reshape(self.test_logprobs, [ntest_batches, test_batch_size, out])

    def get_joint_data(self,
                       key: chex.PRNGKey,
                       nsamples: int,
                       njoint: int
                       ) -> Tuple[Tuple[chex.Array, chex.Array], float]:
        #  https://github.com/deepmind/neural_testbed/blob/7defebf0a232a921d720b870c3ab3a56d38e7ceb/neural_testbed/generative/gp_classification_envlikelihood.py#L106

        def joint_data_per_sample(key):
            out = self.y_test.shape[-1]
            nfeatures = self.X_test.shape[-1]

            X_test = self.X_test.reshape((-1, nfeatures))
            y_test = self.y_test.reshape((-1, out))
            test_logprobs = jnp.reshape(self.test_logprobs, [-1, out])

            x_key, y_key = random.split(key, 2)

            # Sample njoint x's from the testing cache for evaluation.
            test_x_indices = random.randint(x_key, [njoint], 0, len(X_test))

            # For these x indices, find class probabilities.
            test_logprobs = test_logprobs[test_x_indices, :]
            chex.assert_shape(test_logprobs, [njoint, out])

            # For these x indices, find the corresponding x test.
            X_test = X_test[test_x_indices, :]
            chex.assert_shape(X_test, [njoint, nfeatures])

            y_test = y_test[test_x_indices, :]
            chex.assert_shape(y_test, [njoint, 1])

            # Compute the log likelihood with respect to the environment
            log_likelihood = categorical_log_likelihood(test_logprobs, y_test)

            return (X_test, y_test), log_likelihood

        keys = random.split(key, nsamples)
        return vmap(joint_data_per_sample)(keys)