from typing import Optional
import jax.numpy as jnp

import chex
from jsl.experimental.seql.environments.sequential_data_env import SequentialDataEnvironment

from jsl.experimental.seql.utils import gaussian_log_likelihood


class SequentialRegressionEnvironment(SequentialDataEnvironment):
  def __init__(self, 
               X_train: chex.Array,
               y_train: chex.Array,
               X_test: chex.Array,
               y_test: chex.Array,
               ground_truth: chex.Array,
               train_batch_size: int,
               test_batch_size: int,
               obs_noise: float,
               key: Optional[chex.PRNGKey] = None):


    super().__init__(X_train,
                    y_train,
                    X_test,
                    y_test,
                    ground_truth,
                    train_batch_size,
                    test_batch_size,
                    key)
    self.obs_noise = obs_noise

  def reward(self,
             predictions: chex.Array,
             t: int,
             train: bool = False): 
    
    cov = self.obs_noise * jnp.eye(len(predictions))
    if train:
      mu = self.y_train[t] #- predictions
    else:
      mu = self.y_test[t] #- predictions
    return gaussian_log_likelihood(mu, cov, predictions)