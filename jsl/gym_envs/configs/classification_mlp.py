# Default configuration where the problem is regression and the agent is Kalman Filter 
import ml_collections

# Local imports
from configs.utils import PriorKnowledge

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.problem = "classification"
    config.env_model = "MLP"
    config.agent = "KalmanFilter" 
    config.seed = 0
    config.nsteps = 20
    config.ntrials = 20
    config.train_batch_size = 1
    config.test_batch_size = 1

    input_dim, num_train, tau, ouput_dim, hidden = 20, 100, 1, 1,10
    config.prior_knowledge = PriorKnowledge(input_dim, num_train, tau, ouput_dim, hidden=hidden)
    return config