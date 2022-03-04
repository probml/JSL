# Default configuration where the problem is regression and the agent is Kalman Filter 
import ml_collections

from typing import Any, Dict, Optional
import dataclasses


@dataclasses.dataclass(frozen=True)
class PriorKnowledge:
  """What an agent knows a priori about the problem."""
  input_dim: int
  num_train: int
  tau: int
  num_classes: int = 1
  hidden: Optional[int] = None
  noise_std: Optional[float] = None
  temperature: Optional[float] = 1
  extra: Optional[Dict[str, Any]] = None

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.problem = "regression"
    config.env_model = "MLP"
    config.agent = "KalmanFilter" 
    config.seed = 0
    config.nsteps = 20
    config.ntrials = 20
    config.train_batch_size = 1
    config.test_batch_size = 1

    input_dim, num_train, tau, num_classes, hidden = 20, 100, 1, 1,10
    config.prior_knowledge = PriorKnowledge(input_dim, num_train, tau, num_classes, hidden=hidden)
    return config