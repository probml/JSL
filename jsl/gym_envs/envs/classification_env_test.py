"""Tests classifcation_env.py."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk

import jax

import numpy as np

import envs.classification_env as classification_env
from envs.base import PriorKnowledge, make_gaussian_sampler


class MLPClassificationEnsembleTest(parameterized.TestCase):

    @parameterized.parameters(itertools.product([3, 10], [1, 3], [1, 3]))
    def test_valid_data(self, num_steps: int, input_dim: int, tau: int):
        np.random.seed(0)
        num_class = 2
        rng = hk.PRNGSequence(0)
        train_batch_size, test_batch_size = 2, 2

        x_train_generator = lambda k, n: jax.random.normal(k, [n, input_dim])
        x_test_generator = make_gaussian_sampler(
            input_dim)

        fn_transformed = hk.without_apply_rng(hk.transform(
            lambda x: hk.nets.MLP([10, 10, num_class])(x)))  # pylint: disable=[unnecessary-lambda]
        params = fn_transformed.init(next(rng), np.zeros(shape=(input_dim,)))
        logit_fn = lambda x: fn_transformed.apply(params, x)

        prior_knowledge = PriorKnowledge(input_dim, num_steps, tau, num_classes=num_class, hidden=10)
        mlp_model = classification_env.ClassificationEnv(
            apply_fn=logit_fn,
            x_train_generator=x_train_generator,
            x_test_generator=x_test_generator,
            prior_knowledge=prior_knowledge,
            nsteps=2,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            key=next(rng)
        )
        # Check that the training data is reasonable.
        assert mlp_model.x_train.shape[1:] == (train_batch_size, input_dim)
        assert mlp_model.y_train.shape[1:] == (train_batch_size, 1)

        assert np.all(~np.isnan(mlp_model.x_train))
        assert np.all(~np.isnan(mlp_model.y_train))

        # Check that the testing data is reasonable.
        action = jax.numpy.ones((train_batch_size, 1))
        for t in range(2):
            obs, reward, done, info = mlp_model.step(action)
            assert done == False

        # Check that the testing data is reasonable.
        for _ in range(3):
            test_data, log_likelihood = mlp_model.test_data(next(rng))
            x_test, y_test = test_data
            assert np.isfinite(log_likelihood)
            assert x_test.shape == (tau, input_dim)
            assert y_test.shape == (tau, 1)
            assert np.all(~np.isnan(x_test))
            assert np.all(~np.isnan(y_test))

    @parameterized.parameters(itertools.product([1, 10, 100]))
    def test_not_all_test_data_same_x(self, num_steps: int):
        """Generates testing data and checks not all the same x value."""
        np.random.seed(0)
        num_test_seeds = 10
        input_dim = 2
        num_class = 2
        tau = 1
        rng = hk.PRNGSequence(0)

        x_train_generator = lambda k, n: jax.random.normal(k, [n, input_dim])
        x_test_generator = make_gaussian_sampler(
            input_dim)
        fn_transformed = hk.without_apply_rng(hk.transform(
            lambda x: hk.nets.MLP([10, 10, num_class])(x)))  # pylint: disable=[unnecessary-lambda]
        params = fn_transformed.init(next(rng), np.zeros(shape=(input_dim,)))
        logit_fn = lambda x: fn_transformed.apply(params, x)
        train_batch_size, test_batch_size = 1, 1
        prior_knowledge = PriorKnowledge(input_dim, num_steps, tau, num_classes=num_class, hidden=10)

        mlp_model = classification_env.ClassificationEnv(
            apply_fn=logit_fn,
            x_train_generator=x_train_generator,
            x_test_generator=x_test_generator,
            prior_knowledge=prior_knowledge,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            nsteps=num_steps,
            key=next(rng),
        )

        num_distinct_x = 0
        reference_data, _ = mlp_model.test_data(key=next(rng))
        reference_x, _ = reference_data
        for _ in range(num_test_seeds):
            test_data, _ = mlp_model.test_data(key=next(rng))
            x, _ = test_data
            if not np.all(np.isclose(x, reference_x)):
                num_distinct_x += 1
            assert num_distinct_x > 0

    @parameterized.parameters(itertools.product([10], [1], [10]))
    def test_valid_labels(self, num_train: int, input_dim: int, num_seeds: int):
        """Checks that for at most 20% of problems, the labels are degenerate."""
        num_class = 2
        tau = 1
        rng = hk.PRNGSequence(0)

        x_train_generator = lambda k, n: jax.random.normal(k, [n, input_dim])
        x_test_generator = make_gaussian_sampler(
            input_dim)
        fn_transformed = hk.without_apply_rng(hk.transform(
            lambda x: hk.nets.MLP([10, 10, num_class])(x)))  # pylint: disable=[unnecessary-lambda]
        train_batch_size, test_batch_size = 1, 1
        prior_knowledge = PriorKnowledge(input_dim, num_train, tau, num_classes=num_class, hidden=10)

        labels_means = []
        for _ in range(num_seeds):
            params = fn_transformed.init(next(rng), np.zeros(shape=(input_dim,)))
            logit_fn = functools.partial(fn_transformed.apply, params)
            mlp_model = classification_env.ClassificationEnv(
                apply_fn=logit_fn,
                x_train_generator=x_train_generator,
                x_test_generator=x_test_generator,
                prior_knowledge=prior_knowledge,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                nsteps=num_train,
                key=next(rng),
            )

            labels_means.append(np.mean(mlp_model.y_train))

        degenerate_cases = labels_means.count(0.) + labels_means.count(1.)
        # Check that for at most 20% of problems, the labels are degenerate
        assert degenerate_cases / num_seeds <= 0.2


if __name__ == '__main__':
    absltest.main()
