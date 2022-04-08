import chex
import typing_extensions


class MetricFn(typing_extensions.Protocol):
    """Interface for evaluation of multiple posterior samples based on a metric."""

    def __call__(self, predictions: chex.Array, y: chex.Array) -> float:
        """Calculates a metric based on logits and labels.
        Args:
          predictions: An array of shape [A, B, C] where B is the batch size of data, C
            is the number of outputs per data (for classification, this is
            equal to number of classes), and A is the number of random samples for
            each data.
          y: An array of shape [B, 1] where B is the batch size of data.
        Returns:
          A float number specifies the value of the metric.
        """