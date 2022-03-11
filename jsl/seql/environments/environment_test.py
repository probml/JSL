"""Tests for jsl.sent.environments.base"""
import torchvision

from jax import random

import itertools
from typing import List, Tuple

from absl.testing import absltest
from absl.testing import parameterized

from jsl.seql.environments import base
from jsl.seql.environments.sequential_data_env import SequentialDataEnvironment

class EnvironmentTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product([3, 4], [(1, 10), (2, 20)]))
  def test_gaussian_sampler(self, seed: int, shape: Tuple):
    key = random.PRNGKey(seed)
    x = base.gaussian_sampler(key, shape)
    self.assertEqual(x.shape, shape)
  

  @parameterized.parameters(itertools.product([10.], [1, 10], [False, True]))
  def test_evenly_spaced_sampler(self,
                                max_val: float, num_samples: int,
                                use_bias: bool):
    x = base.eveny_spaced_x_sampler(max_val, num_samples, use_bias)
    
    if use_bias:
      nfeatures = 2
      self.assertEqual(x.shape, (num_samples, nfeatures))
    else:
      nfeatures = 1
      self.assertEqual(x.shape, (num_samples, nfeatures))


  def _check_seq_data_env_params(self,
                                env: SequentialDataEnvironment,
                                input_dim: int, output_dim: int,
                                ntrain: int, ntest: int,
                                train_batch_size: int,
                                test_batch_size: int):
    
    ntrain_batches = ntrain // train_batch_size
    ntest_batches = ntest // test_batch_size

    # Checks training data is correct
    self.assertEqual(env.y_train.shape, (ntrain_batches, train_batch_size, output_dim))
    self.assertEqual(env.X_train.shape, (ntrain_batches, train_batch_size, input_dim))

    # Checks test data is correct
    self.assertEqual(env.y_test.shape, (ntest_batches, test_batch_size, output_dim))
    self.assertEqual(env.X_test.shape, (ntest_batches, test_batch_size, input_dim))

  
  @parameterized.parameters(itertools.product([0], [1, 3], [10], [10],  [1, 2], [1, 2]))
  def test_make_random_poly_regression_environment(self,
                                                  seed: int,
                                                  degree: int,
                                                  ntrain, ntest,
                                                  train_batch_size: int,
                                                  test_batch_size: int):

    key = random.PRNGKey(seed)

    env = base.make_random_poly_regression_environment(key,
                                                  degree,
                                                  ntrain, ntest,
                                                  train_batch_size=train_batch_size,
                                                  test_batch_size=test_batch_size)

    output_dim = 1
    self._check_seq_data_env_params(env, degree + 1, output_dim,
                                    ntrain, ntest, train_batch_size,
                                    test_batch_size)
  

  @parameterized.parameters(itertools.product([0], [1, 10], [1, 10], [10],
                                              [10], [0., 0.5], [1, 2], [1, 2]))
  def test_make_random_linear_regression_environment(self,
                                                    seed: int,
                                                    nfeatures: int,
                                                    ntargets: int,
                                                    ntrain: int,
                                                    ntest: int,
                                                    bias: float,
                                                    train_batch_size: int,
                                                    test_batch_size: int):
    
    key = random.PRNGKey(seed)
    
    env = base.make_random_linear_regression_environment(key,
                                                    nfeatures,
                                                    ntargets,
                                                    ntrain,
                                                    ntest,
                                                    bias,
                                                    train_batch_size=train_batch_size,
                                                    test_batch_size=test_batch_size)
    self._check_seq_data_env_params(env, nfeatures, ntargets,
                                    ntrain, ntest, train_batch_size,
                                    test_batch_size)

  @parameterized.parameters(itertools.product([0], [1, 10], [1, 10], [20],
                           [20], [1., 2.], [[10], [2, 2]], [1, 2], [1,2]))
  def test_make_classification_mlp_environment(self, seed: int,
                                              nfeatures: int,
                                              ntargets: int,
                                              ntrain: int,
                                              ntest: int,
                                              temperature: float,
                                              hidden_layer_sizes: List[int],
                                              train_batch_size,
                                              test_batch_size):

    key = random.PRNGKey(seed)
    
    env = base.make_classification_mlp_environment(key,
                                                  nfeatures,
                                                  ntargets,
                                                  ntrain,
                                                  ntest,
                                                  temperature,
                                                  hidden_layer_sizes,
                                                  train_batch_size=train_batch_size,
                                                  test_batch_size=test_batch_size)
    output_dim = 1
    self._check_seq_data_env_params(env, nfeatures, output_dim,
                                    ntrain, ntest, train_batch_size,
                                    test_batch_size)
    
    self.assertEqual(env.y_train.max(), ntargets-1)
    self.assertEqual(env.y_train.min(), 0)

    self.assertEqual(env.y_test.max(), ntargets-1)
    self.assertEqual(env.y_test.min(), 0)
  
  @parameterized.parameters(itertools.product([0], [1, 10], [1, 10],
                          [20], [20], [1., 2.], [[10], [2, 2]], [1, 2], [1,2]))
  def test_make_regression_mlp_environment(self,
                                    seed:int,
                                    nfeatures: int,
                                    ntargets: int,
                                    ntrain: int,
                                    ntest: int,
                                    temperature: float,
                                    hidden_layer_sizes: List[int],
                                    train_batch_size: int,
                                    test_batch_size: int):
    key = random.PRNGKey(seed)
    
    env = base.make_regression_mlp_environment(key,
                                              nfeatures,
                                              ntargets,
                                              ntrain,
                                              ntest,
                                              temperature,
                                              hidden_layer_sizes,
                                              train_batch_size,
                                              test_batch_size)
    self._check_seq_data_env_params(env, nfeatures, ntargets,
                                    ntrain, ntest, train_batch_size,
                                    test_batch_size)


  @parameterized.parameters(itertools.product([1, 20], [1, 20]))
  def test_make_environment_from_torch_dataset(self,
                                    train_batch_size: int,
                                    test_batch_size: int):
    dataset = torchvision.datasets.MNIST
    classification = True
    env = base.make_environment_from_torch_dataset(dataset,
                                                  classification,
                                                  train_batch_size,
                                                  test_batch_size)
    X_train, y_train, X_test, y_test = env.get_data(0)
    
    self.assertEqual(X_train.shape[0], train_batch_size)
    self.assertEqual(y_train.shape[0], train_batch_size)
    self.assertEqual(X_test.shape[0], test_batch_size)
    self.assertEqual(y_test.shape[0], test_batch_size)

if __name__ == '__main__':
  absltest.main()