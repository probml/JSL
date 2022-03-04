# Default configuration where the problem is regression and the agent is Kalman Filter 
import ml_collections
import jax.numpy as jnp

# Local imports
from configs.utils import AgentConfig, EnvironmentConfig, PriorKnowledge

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.nsteps = 50
    config.ntrials = 1

    problem_type = "regression"
    model = "linear"

    input_dim = 2
    tau = 1
    output_dim = 1
    prior_knowledge = PriorKnowledge(input_dim, tau, output_dim=output_dim)

    # Random seed controlling all the randomness in the problem.
    config.seed = 0

    config.env = EnvironmentConfig(problem_type, model, prior_knowledge)

    model = "KalmanFilter"
    
    init_kwargs = dict(mu0=jnp.zeros(input_dim),
                        Sigma0=jnp.eye(input_dim) * 10.,
                        F=jnp.eye(input_dim),
                        Q=0,
                        R=1)

    config.agent = AgentConfig(model, init_kwargs=init_kwargs)

    return config
