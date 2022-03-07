"""
Regression-based gym environment.
"""
import jax.numpy as jnp
from jax import jit

import haiku as hk
import chex

from gym import Env, spaces
from typing import Callable, Any

from jsl.gym_envs.envs.base import mean_squared_error, sample_gaussian_reg_data

class RegressionEnv(Env):

  def __init__(self,
              apply_fn: Callable,
              x_train_generator: Callable,
              x_test_generator: Callable,
              prior_knowledge: Any,
              train_batch_size: int,
              test_batch_size:int,  
              nsteps:  int,
              key: chex.PRNGKey,
              sample_fn: Callable = sample_gaussian_reg_data):

    super(RegressionEnv, self).__init__()

    # Key sequences
    self.rng = hk.PRNGSequence(key)

    self.apply_fn = apply_fn
    self.x_train_generator = x_train_generator
    self.x_test_generator = x_test_generator
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.nsteps = nsteps
    
    if sample_fn is None:
      self.sample_fn = sample_gaussian_reg_data
    else:
      self.sample_fn = sample_fn

    self.tau = prior_knowledge.tau
    self.input_dim = prior_knowledge.input_dim

    self.t = 0

    # Environment OpenAI metadata
    self.reward_range = spaces.Box(low=-jnp.inf, high=0., 
                                   shape=(train_batch_size, 1), dtype=jnp.float32)
    self.action_space = spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                   shape=(train_batch_size, 1), dtype=jnp.float32)
    self.observation_space = {
                              "X_train":spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(train_batch_size, self.input_dim), dtype=jnp.float32),
                              "Y_train":spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(train_batch_size, 1), dtype=jnp.float32),
                              "X_test": spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(test_batch_size, self.input_dim), dtype=jnp.float32),
                              "Y_test": spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(test_batch_size, 1), dtype=jnp.float32)
                            }


  def _initialize_data(self):
    nsamples = self.nsteps * self.train_batch_size
    (x_train, y_train) = self.sample_fn(
        self.apply_fn, self.x_train_generator,
        nsamples, next(self.rng))
    self.x_train = x_train.reshape((-1, self.train_batch_size, self.input_dim))
    self.y_train =  y_train.reshape((-1, self.train_batch_size, 1))
    (x_test, y_test) = self.sample_fn(
            self.apply_fn, self.x_train_generator,
            nsamples, next(self.rng))
    self.x_test = x_test.reshape((-1, self.test_batch_size, self.input_dim))
    self.y_test =  y_test.reshape((-1, self.test_batch_size, 1))

  def reset(self):
    self._initialize_data()
    self.t = 0
    # Returns current_state
    return {"X_train": self.x_train[self.t],
            "y_train": self.y_train[self.t],
            "X_test": self.x_test[self.t],
            "y_test": self.y_test[self.t]}

  @property
  def done(self):
    return  self.t >= self.x_train.shape[0]

  def step(self, action):
    done = self.done
    info = {}
    
    y = self.y_train[self.t]
    reward = -mean_squared_error(action, y)
    self.t += 1
    
    if done:
      observation = {}
    else:
      observation = { "X_train": self.x_train[self.t],
                      "y_train": self.y_train[self.t],
                      "X_test": self.x_test[self.t],
                      "y_test": self.y_test[self.t]
                      }
    return observation, reward, done, info


  def test_data(self, key: chex.PRNGKey):
    """Generates test data and evaluates log likelihood w.r.t. environment.
    The test data that is output will be of length tau examples.
    We wanted to "pass" tau here... but ran into jit issues.
    Args:
      key: Random number generator key.
    Returns:
      Tuple of data (with tau examples) and log-likelihood under posterior.
    """
    def sample_test(k: chex.PRNGKey):
      (x_train, y_train), _, _ =  sample_gaussian_reg_data(
          self.logit_fn, self.x_test_generator, self.tau, key=k)
      return x_train, y_train

    return jit(sample_test)(key)


  def render(self):
    pass

  @property
  def done(self):
      return self.t >= self.x_train.shape[0]

  def step(self, action):
      done = self.done
      info = {}

      y = self.y_train[self.t]
      err = mean_squared_error(action, y)
      reward = -mean_squared_error(action, y)
      self.t += 1

      if done:
          observation = {}
      else:
          observation = {"X_train": self.x_train[self.t],
                          "y_train": self.y_train[self.t],
                          "X_test": self.x_test[self.t],
                          "y_test": self.y_test[self.t]
                          }
      return observation, reward, done, info

  def test_data(self, key: chex.PRNGKey):
      """Generates test data and evaluates log likelihood w.r.t. environment.
      The test data that is output will be of length tau examples.
      We wanted to "pass" tau here... but ran into jit issues.
      Args:
        key: Random number generator key.
      Returns:
        Tuple of data (with tau examples) and log-likelihood under posterior.
      """

      def sample_test(k: chex.PRNGKey):
          (x_train, y_train), _, _ = sample_gaussian_reg_data(
              self.logit_fn, self.x_test_generator, self.tau, key=k)
          return x_train, y_train

      return jit(sample_test)(key)

  def render(self):
      pass
