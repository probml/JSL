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
               ground_truth: chex.Array, 
               train_batch_size: int,
               test_batch_size: int,
               key: Optional[chex.PRNGKey] = None):
    
    self.ground_truth = ground_truth 

    ntrain, nfeatures = X_train.shape
    ntest, out = y_train.shape

    # TODO: It will produce an error if ntrain % train_batch_size != 0
    ntrain_batches = ntrain // train_batch_size
    # TODO: It will produce an error if ntest % test_batch_size != 0
    ntest_batches = ntest // test_batch_size

    self.X_train = jnp.reshape(X_train, [ntrain_batches, train_batch_size, nfeatures])
    self.y_train = jnp.reshape(y_train, [ntrain_batches, train_batch_size, out])

    self.X_test = jnp.reshape(X_test, [ntest_batches, test_batch_size, nfeatures])
    self.y_test =jnp.reshape(y_test, [ntest_batches, test_batch_size, out])
    
    self.train_indices = jnp.arange(ntrain)
    self.test_indices = jnp.arange(ntest)

    if key is not None:
      self.shuffle_data(key)

  def get_data(self, t: int):
    return self.X_train[t], self.y_train[t], self.X_test[t], self.y_test[t]

  def reward(self,
             predictions: chex.Array, 
             t: int,
             train: bool = False): 
    pass
  
  def shuffle_data(self, key: chex.PRNGKey):
    train_key, test_key = random.split(key)

    train_indices = random.permutation(train_key, self.train_indices)
    self.X_train = self.X_train[train_indices]
    self.y_train = self.y_train[train_indices]

    test_indices = random.permutation(test_key, self.test_indices)
    self.X_test = self.X_test[test_indices]
    self.y_test = self.y_test[test_indices]

  def reset(self, key: chex.PRNGKey):
    
    ntrain_batches, train_batch_size, nfeatures = self.X_train.shape
    ntest_batches, test_batch_size, out = self.y_test.shape

    self.shuffle_data(key)

    self.X_train = jnp.reshape(self.X_train, [ntrain_batches, train_batch_size, nfeatures])
    self.y_train = jnp.reshape(self.y_train, [ntrain_batches, train_batch_size, out])

    self.X_test = jnp.reshape(self.X_test, [ntest_batches, test_batch_size, nfeatures])
    self.y_test =jnp.reshape(self.y_test, [ntest_batches, test_batch_size, out])