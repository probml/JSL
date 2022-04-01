from typing import Optional
import jax.numpy as jnp
from jax import nn

import chex
from jsl.experimental.seql.environments.sequential_data_env import SequentialDataEnvironment

from jsl.experimental.seql.utils import categorical_log_likelihood


class SequentialClassificationEnvironment(SequentialDataEnvironment):

  def __init__(self, 
               X_train: chex.Array,
               y_train: chex.Array,
               X_test: chex.Array,
               y_test: chex.Array,
               ground_truth: chex.Array,
               train_batch_size: int,
               test_batch_size: int,
               logprobs: chex.Array,
               key: Optional[chex.PRNGKey] = None):

    super().__init__(X_train,
                    y_train,
                    X_test,
                    y_test,
                    ground_truth,
                    train_batch_size,
                    test_batch_size,
                    key)

    ntrain = len(y_train)
    ntest = len(y_test)
    _, out = logprobs.shape

    ntrain_batches = ntrain // train_batch_size
    ntest_batches = ntest // test_batch_size

    self.train_logprobs =jnp.reshape(logprobs[:ntrain], [ntrain_batches, train_batch_size, out])
    self.test_logprobs =jnp.reshape(logprobs[ntrain:], [ntest_batches, test_batch_size, out])

  def reward(self,
             predictions: chex.Array, 
             t: int,
             train: bool = False): 
    labels = jnp.argmax(nn.log_softmax(predictions, axis=-1), axis=-1)
    if train:
      return categorical_log_likelihood(self.train_logprobs[t], labels)
    else:
      return categorical_log_likelihood(self.test_logprobs[t], labels)

  def shuffle_data(self, key: chex.PRNGKey):
    super().shuffle_data(key)
    self.train_logprobs = self.logprobs[self.train_indices]
    self.test_logprobs = self.logprobs[self.test_indices]