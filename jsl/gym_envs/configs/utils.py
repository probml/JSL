import dataclasses
from typing import Optional, Dict, Any, Callable

# Local imports
from jsl.gym_envs.envs.base import make_gaussian_sampler


@dataclasses.dataclass(frozen=True)
class PriorKnowledge:
  """What an agent knows a priori about the problem."""
  # Number of features
  input_dim: int
  tau: int
  output_dim: int = 1
  hidden_layer_sizes: Optional[int] = None
  noise_std: Optional[float] = None
  temperature: Optional[float] = 1
  extra: Optional[Dict[str, Any]] = None

@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
  """Problem configuration including prior knowledge and some hyperparams."""
  # Type of supervised learning problem.
  problem_type: str
  # Model apply function of which is used to obtain the ground-truth y values.
  model: str 
  # Agent's a priori knowledge about the problem.
  prior_knowledge: PriorKnowledge
  # Test sampling distribution
  train_distribution: Callable = make_gaussian_sampler
  # Test sampling distribution
  test_distribution: Callable = make_gaussian_sampler
  # Data sampling function
  sample_fn: Callable = None
  # Number of inputs (X's) used for evaluation.
  num_test_seeds: int = 1000

  train_batch_size: int = 1
  test_batch_size: int = 1

  @property
  def meta_data(self):
    meta = dataclasses.asdict(self)
    meta.pop('prior_knowledge')
    meta.update(dataclasses.asdict(self.prior_knowledge))
    return meta

@dataclasses.dataclass(frozen=True)
class AgentConfig:
    model: str
    init_kwargs: Dict[str, Any]

@dataclasses.dataclass(frozen=True)
class ShiftConfig:
  """Configuration for distributional shift of input data."""
  reject_prob: float
  fraction_rejected_classes: float
