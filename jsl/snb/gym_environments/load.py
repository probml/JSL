"""Loading a leaderboard instance for the testbed."""

from typing import Optional, Tuple

import chex
import haiku as hk
import jax



def problem_from_config(problem_config):
  """Returns a testbed problem given a problem config."""
  assert problem_config.prior_knowledge.num_classes > 0

  if problem_config.prior_knowledge.num_classes > 1:
    return load_classification(problem_config)
  else:
    return load_regression(problem_config)


def problem_with_distribution_shift(
    problem_config, shift_config):
  """Returns a classification problem with input distribution shift."""
  return load_classification(problem_config, shift_config)


def load_classification(
    problem_config,
    shift_config= None,
):
  """Loads a classification problem from problem_config, optional shift_config."""
  
  # Defines the sequence of rng keys
  rng = hk.PRNGSequence(problem_config.seed)
  
  # Prior knowledge consisting of specific info about the problem
  prior_knowledge = problem_config.prior_knowledge
  
  # Number of features
  input_dim = prior_knowledge.input_dim

  logit_fn = generative.make_2layer_mlp_logit_fn(
      input_dim=input_dim,
      temperature=prior_knowledge.temperature,
      hidden=50,
      num_classes=prior_knowledge.num_classes,
      key=next(rng),
  )
  
  if shift_config is None:
    override_train_data = None
  else:
    override_train_data = generative.make_filtered_gaussian_data(
        input_dim=prior_knowledge.input_dim,
        logit_fn=logit_fn,
        reject_prob=shift_config.reject_prob,
        fraction_rejected_classes=shift_config.fraction_rejected_classes,
        num_samples=prior_knowledge.num_train,
        key=next(rng),
    )
  data_sampler = generative.ClassificationEnvLikelihood(
      logit_fn=logit_fn,
      x_train_generator=generative.make_gaussian_sampler(input_dim),
      x_test_generator=problem_config.test_distribution(input_dim),
      num_train=prior_knowledge.num_train,
      key=next(rng),
      override_train_data=override_train_data,
      tau=prior_knowledge.tau,
  )
  return likelihood.SampleBasedTestbed(
      data_sampler=data_sampler,
      sample_based_kl=make_categorical_kl_estimator(problem_config, next(rng)),
      prior_knowledge=prior_knowledge,
  )


def make_categorical_kl_estimator(
    problem_config,
    key: chex.PRNGKey):
  """Make sample based KL estimator for categorial models."""
  prior_knowledge = problem_config.prior_knowledge
  if prior_knowledge.tau > 10:
    sample_based_kl = likelihood.CategoricalClusterKL(
        cluster_alg=likelihood.RandomProjection(dimension=7),
        num_enn_samples=problem_config.num_enn_samples,
        num_test_seeds=problem_config.num_test_seeds,
        key=key,
    )
  else:
    sample_based_kl = likelihood.CategoricalKLSampledXSampledY(
        num_test_seeds=problem_config.num_test_seeds,
        num_enn_samples=problem_config.num_enn_samples,
        key=key,
        num_classes=prior_knowledge.num_classes,
    )
  sample_based_kl = likelihood.add_classification_accuracy_ece(
      sample_based_kl,
      num_test_seeds=int(1_000 / prior_knowledge.tau) + 1,
      num_enn_samples=100,
      num_classes=prior_knowledge.num_classes,
  )
  return sample_based_kl


def gaussian_data(key: chex.PRNGKey,
                  num_train: int,
                  input_dim: int,
                  num_test: int) -> Tuple[chex.Array, chex.Array]:
  """Generate Gaussian training and test data."""
  train_key, test_key = jax.random.split(key)
  x_train = jax.random.normal(train_key, [num_train, input_dim])
  x_test = jax.random.normal(test_key, [num_test, input_dim])
  return x_train, x_test


def load_regression(
    problem_config):
  """Loads a regression problem from problem_config."""
  rng = hk.PRNGSequence(problem_config.seed)
  prior_knowledge = problem_config.prior_knowledge

  x_train, x_test = gaussian_data(
      key=next(rng),
      num_train=prior_knowledge.num_train,
      input_dim=prior_knowledge.input_dim,
      num_test=problem_config.num_test_cache,
  )

  if problem_config.epistemic_only:
    # Special case used only for the ENN paper.
    assert prior_knowledge.tau == 1, 'Only works for tau=1'
    data_sampler = generative.GPRegression(
        kernel_fn=generative.make_benchmark_kernel(prior_knowledge.input_dim),
        x_train=x_train,
        x_test=x_test,
        key=next(rng),
        tau=prior_knowledge.tau,
        noise_std=prior_knowledge.noise_std,
    )
    return generative.TestbedGPRegression(
        data_sampler,
        prior_knowledge,
        key=next(rng),
        num_enn_samples=problem_config.num_enn_samples)

  data_sampler = generative.GPRegressionEnvLikelihood(
      kernel_fn=generative.make_benchmark_kernel(prior_knowledge.input_dim),
      x_train=x_train,
      x_test=x_test,
      key=next(rng),
      tau=prior_knowledge.tau,
      noise_std=prior_knowledge.noise_std,
  )
  sample_based_kl = likelihood.GaussianSampleKL(
      # This KL estimator cannot handle very large num_test_seed * tau
      num_test_seeds=int(problem_config.num_test_seeds
                         / prior_knowledge.tau) + 1,
      num_enn_samples=problem_config.num_enn_samples,
      enn_sigma=prior_knowledge.noise_std,
      key=next(rng),
  )
  return likelihood.SampleBasedTestbed(
      data_sampler=data_sampler,
      sample_based_kl=sample_based_kl,
      prior_knowledge=prior_knowledge,
  )
