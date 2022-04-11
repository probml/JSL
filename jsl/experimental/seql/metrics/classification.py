import dataclasses

import jax.numpy as jnp
from jax import nn, vmap

import chex
from typing import Tuple

from tensorflow_probability.substrates import jax as tfp

from jsl.experimental.seql.metrics.base import MetricFn
from jsl.experimental.seql.metrics.utils import average_sampled_log_likelihood
from jsl.experimental.seql.utils import categorical_log_likelihood


@dataclasses.dataclass
class CalibrationErrorCalculator(MetricFn):
    """Computes expected calibration error (ece) aggregated over enn samples."""
    # https://github.com/deepmind/neural_testbed/blob/7defebf0a232a921d720b870c3ab3a56d38e7ceb/neural_testbed/likelihood/classification.py#L88
    num_bins: int
    name: str = "ece"

    def __call__(self,
                 predictions: chex.Array,
                 test_data: Tuple[Tuple[chex.Array, chex.Array], float]) -> float:
        """Returns ece."""
        (_, y), _ = test_data
        chex.assert_rank(predictions, 3)
        unused_num_enn_samples, num_data, num_classes = predictions.shape
        chex.assert_shape(y, [num_data, 1])

        class_probs = nn.softmax(predictions)
        mean_class_prob = jnp.mean(class_probs, axis=0)
        chex.assert_shape(mean_class_prob, [num_data, num_classes])

        predictions = jnp.argmax(mean_class_prob, axis=1)[:, None]
        chex.assert_shape(predictions, y.shape)

        # ece
        mean_class_logits = jnp.log(mean_class_prob)
        chex.assert_shape(mean_class_logits, (num_data, num_classes))
        labels_true = jnp.squeeze(y, axis=-1)
        chex.assert_shape(labels_true, (num_data,))
        labels_predicted = jnp.squeeze(predictions, axis=-1)
        chex.assert_shape(labels_predicted, (num_data,))
        return tfp.stats.expected_calibration_error(
            num_bins=self.num_bins,
            logits=mean_class_logits,
            labels_true=labels_true,
            labels_predicted=labels_predicted,
        )


@dataclasses.dataclass
class AccuracyCalculator(MetricFn):
    """Computes classification accuracy (acc) aggregated over enn samples."""
    name: str = "accuracy"

    def __call__(self,
                 predictions: chex.Array,
                 test_data: Tuple[Tuple[chex.Array, chex.Array], float]) -> float:
        (_, y), _ = test_data
        # https://github.com/deepmind/neural_testbed/blob/7defebf0a232a921d720b870c3ab3a56d38e7ceb/neural_testbed/likelihood/classification.py#L120
        chex.assert_rank(predictions, 3)
        unused_num_enn_samples, num_data, num_classes = predictions.shape
        chex.assert_shape(y, [num_data, 1])

        class_probs = nn.softmax(predictions)
        mean_class_prob = jnp.mean(class_probs, axis=0)
        chex.assert_shape(mean_class_prob, [num_data, num_classes])

        predictions = jnp.argmax(mean_class_prob, axis=1)[:, None]
        chex.assert_shape(predictions, [num_data, 1])

        return jnp.mean(predictions == y)


@dataclasses.dataclass
class JointLogLikelihoodCalculator(MetricFn):
    name: str = "ll"

    def __call__(self,
                 predictions: chex.Array,
                 test_data: Tuple[Tuple[chex.Array, chex.Array], float]) -> float:
        """Computes joint log likelihood (ll) aggregated over enn samples.
        Depending on data batch_size (can be inferred from logits and labels), this
        function computes joint ll for tau=batch_size aggregated over enn samples. If
        num_data is one, this function computes marginal ll.
        Args:
          logits: [num_enn_sample, num_data, num_classes]
          labels: [num_data, 1]
        Returns:
          marginal log likelihood
        """
        (_, y), _ = test_data
        num_enn_samples, tau, num_classes = predictions.shape
        chex.assert_shape(y, (tau, 1))

        class_probs = nn.softmax(predictions)
        chex.assert_shape(class_probs, (num_enn_samples, tau, num_classes))

        batched_ll = vmap(categorical_log_likelihood, in_axes=[0, None])
        sampled_ll = batched_ll(class_probs, y)
        return average_sampled_log_likelihood(sampled_ll)


@dataclasses.dataclass
class KLEstimationCalculator(MetricFn):
    """Evaluates KL according to categorical model, sampling X and output Y.
    This approach samples an (x, y) output from the enn and data sampler and uses
    this to estimate the KL divergence.
    """
    name: str = "kl"

    def __call__(self,
                 predictions: chex.Array,
                 test_data: Tuple[Tuple[chex.Array, chex.Array], float]) -> float:
        """Evaluates KL according to categorical model."""
        num_enn_samples, tau, num_classes = predictions.shape
        true_ll = test_data[-1]
        return true_ll - JointLogLikelihoodCalculator(predictions, test_data)
