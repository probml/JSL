"""
Classification-based gym environment
It is based on 
https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/generative/classification_envlikelihood.py
"""

import jax.numpy as jnp

import chex

from jax import jit

import haiku as hk

from gym import Env, spaces
from typing import Callable, Any

from jsl.gym_envs.envs.base import sample_gaussian_cls_data, categorical_log_likelihood

class ClassificationEnv(Env):

<<<<<<< HEAD
  def __init__(self,
              apply_fn: Callable,
              x_train_generator: Callable,
              x_test_generator: Callable,
              prior_knowledge: Any,
              train_batch_size: int,
              test_batch_size:int,  
              nsteps:  int,
              key: chex.PRNGKey,
              sample_fn: Callable = sample_gaussian_cls_data):

    super(ClassificationEnv, self).__init__()
  
    # Key sequences
    self.rng = hk.PRNGSequence(key)

    self.apply_fn = apply_fn
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.x_train_generator = x_train_generator
    self.x_test_generator = x_test_generator
    self.nsteps = nsteps
    
    if sample_fn is None:
      self.sample_fn = sample_gaussian_cls_data
    else:
      self.sample_fn = sample_fn
    
    self.tau = prior_knowledge.tau
    self.input_dim = prior_knowledge.input_dim

    self.t = 0

    # Environment OpenAI metadata
    self.reward_range = spaces.Discrete(prior_knowledge.output_dim)
    self.action_space = spaces.MultiDiscrete([prior_knowledge.output_dim] * train_batch_size)
    self.observation_space = {
                              "X_train":spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(train_batch_size, self.input_dim), dtype=jnp.float32),
                              "y_train":spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(train_batch_size, 1), dtype=jnp.float32),
                              "X_test": spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(test_batch_size, self.input_dim), dtype=jnp.float32),
                              "y_test": spaces.Box(low=-jnp.inf, high=jnp.inf, 
                                        shape=(test_batch_size, 1), dtype=jnp.float32)
                            }


  @property
  def done(self):
    return  self.t >= self.x_train.shape[0]

  def step(self, action):
    done = self.done
    info = {}

    reward = -categorical_log_likelihood(self.train_probs, action)
    self.t += 1
    
    if done:
      observation = {}
    else:
      observation = { 
                      "X_train": self.x_train[self.t],
                      "y_train": self.y_train[self.t],
                      "X_test": self.x_test[self.t],
                      "y_test": self.y_test[self.t]
                    }
    return observation, reward, done, info

  def _initialize_data(self):
    nsamples = self.nsteps * self.train_batch_size
    (x_train, y_train), train_probs,  _ = self.sample_fn(
            self.apply_fn, self.x_train_generator,
            nsamples, next(self.rng))
    self.x_train = x_train.reshape((-1, self.train_batch_size, self.input_dim))
    self.train_probs = train_probs
    self.y_train =  y_train.reshape((-1, self.train_batch_size, 1))

    (x_test, y_test), train_probs, _ = self.sample_fn(
                self.apply_fn, self.x_train_generator,
                nsamples, next(self.rng))
        
    self.x_test = x_test.reshape((-1, self.test_batch_size, self.input_dim))
    self.y_test =  y_test.reshape((-1, self.test_batch_size, 1))

  def reset(self):
      self._initialize_data()
      self.t = 0
      # Returns the current state
      return {"X_train": self.x_train[self.t],
            "y_train": self.y_train[self.t],
            "X_test": self.x_test[self.t],
            "y_test": self.y_test[self.t]}


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
      (x_train, y_train), _, log_likelihood =  sample_gaussian_cls_data(
          self.apply_fn, self.x_test_generator, self.tau, key=k)
      return (x_train, y_train), log_likelihood

    return jit(sample_test)(key)

  def render(self):
    pass
=======
    def __init__(self,
                 logit_fn: Callable,
                 x_train_generator: Callable,
                 x_test_generator: Callable,
                 prior_knowledge: PriorKnowledge,
                 train_batch_size: int,
                 test_batch_size: int,
                 num_steps: int,
                 key: chex.PRNGKey):

        assert prior_knowledge.num_classes > 1

        # Key sequences
        rng = hk.PRNGSequence(key)

        self.logit_fn = logit_fn
        self.tau = prior_knowledge.tau
        self.x_test_generator = x_test_generator
        self.num_steps = num_steps

        input_dim = prior_knowledge.input_dim

        # Optionally override training data where you want to allow for training
        # data that was *not* generated by the x_generator, logit_fn.
        nsamples = num_steps * train_batch_size
        (x_train, y_train), train_probs, _ = sample_gaussian_data(
            logit_fn, x_train_generator,
            nsamples, next(rng))
        self.x_train = x_train.reshape((-1, train_batch_size, input_dim))
        self.train_probs = train_probs
        self.y_train = y_train.reshape((-1, train_batch_size, 1))

        (x_test, y_test), train_probs, _ = sample_gaussian_data(
            logit_fn, x_train_generator,
            nsamples, next(rng))

        self.x_test = x_test.reshape((-1, test_batch_size, input_dim))
        self.y_test = y_test.reshape((-1, test_batch_size, 1))

        # Environment OpenAI metadata
        self.reward_range = spaces.Box(low=-jnp.inf, high=1., dtype=jnp.float32)
        self.action_space = spaces.MultiDiscrete([prior_knowledge.num_classes] * train_batch_size)
        self.observation_space = {
            "X_train": spaces.Box(low=-jnp.inf, high=jnp.inf,
                                  shape=(train_batch_size, input_dim), dtype=jnp.float32),
            "y_train": spaces.Box(low=-jnp.inf, high=jnp.inf,
                                  shape=(train_batch_size, 1), dtype=jnp.float32),
            "X_test": spaces.Box(low=-jnp.inf, high=jnp.inf,
                                 shape=(test_batch_size, input_dim), dtype=jnp.float32),
            "y_test": spaces.Box(low=-jnp.inf, high=jnp.inf,
                                 shape=(test_batch_size, 1), dtype=jnp.float32)
        }
        self.t = 0
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    @property
    def done(self):
        return self.t >= self.x_train.shape[0]

    def step(self, action):
        done = self.done
        info = {}

        reward = -categorical_log_likelihood(self.train_probs, action)
        self.t += 1

        if done:
            observation = {}
        else:
            observation = {
                "X_train": self.x_train[self.t],
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
            (x_train, y_train), _, log_likelihood = base.sample_gaussian_data(
                self.logit_fn, self.x_test_generator, self.tau, key=k)
            return (x_train, y_train), log_likelihood

        return jit(sample_test)(key)

    def render(self):
        pass
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c
