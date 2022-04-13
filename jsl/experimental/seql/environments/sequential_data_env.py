from typing import Optional
import jax.numpy as jnp
from jax import random

import chex


class SequentialDataEnvironment:
    def __init__(self,
                 X_train: chex.Array,
                 y_train: chex.Array,
                 X_test: chex.Array,
                 y_test: chex.Array,
                 true_model: chex.Array,
                 train_batch_size: int,
                 test_batch_size: int,
                 key: Optional[chex.PRNGKey] = None):
        self.true_model = true_model

        ntrain, nfeatures = X_train.shape
        ntest, out = y_test.shape

        self.train_indices = jnp.arange(ntrain)
        self.test_indices = jnp.arange(ntest)

        # TODO: It will produce an error if ntrain % train_batch_size != 0
        ntrain_batches = ntrain // train_batch_size
        # TODO: It will produce an error if ntest % test_batch_size != 0
        ntest_batches = ntest // test_batch_size

        self.X_train = jnp.reshape(X_train, [ntrain_batches, train_batch_size, nfeatures])
        self.y_train = jnp.reshape(y_train, [ntrain_batches, train_batch_size, out])

        self.X_test = jnp.reshape(X_test, [ntest_batches, test_batch_size, nfeatures])
        self.y_test = jnp.reshape(y_test, [ntest_batches, test_batch_size, out])

        if key is not None:
            self.shuffle_data(key)

    def get_data(self, t: int):
        return self.X_train[t], self.y_train[t], self.X_test[t], self.y_test[t]

    def shuffle_data(self, key: chex.PRNGKey):
        ntrain_batches, train_batch_size, nfeatures = self.X_train.shape
        ntest_batches, test_batch_size, out = self.y_test.shape

        self.X_train = jnp.reshape(self.X_train, [-1, nfeatures])
        self.y_train = jnp.reshape(self.y_train, [-1, out])

        self.X_test = jnp.reshape(self.X_test, [-1, nfeatures])
        self.y_test = jnp.reshape(self.y_test, [-1, out])

        train_key, test_key = random.split(key)

        self.train_indices = random.permutation(train_key, self.train_indices)
        self.X_train = self.X_train[self.train_indices]
        self.y_train = self.y_train[self.train_indices]

        self.test_indices = random.permutation(test_key, self.test_indices)
        self.X_test = self.X_test[self.test_indices]
        self.y_test = self.y_test[self.test_indices]

        self.X_train = jnp.reshape(self.X_train, [ntrain_batches, train_batch_size, nfeatures])
        self.y_train = jnp.reshape(self.y_train, [ntrain_batches, train_batch_size, out])

        self.X_test = jnp.reshape(self.X_test, [ntest_batches, test_batch_size, nfeatures])
        self.y_test = jnp.reshape(self.y_test, [ntest_batches, test_batch_size, out])

    def reset(self, key: chex.PRNGKey):
        self.shuffle_data(key)
