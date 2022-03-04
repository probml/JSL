import jax.numpy as jnp
import ml_collections
import chex

from typing import Callable

from jsl.gym_envs.configs.utils import PriorKnowledge, EnvironmentConfig, AgentConfig

def make_evenly_spaced_x_sampler(input_dim: int):
    def make_evenly_spaced_x(key: chex.PRNGKey, num_samples: int= 21, use_bias=True):
        X = jnp.linspace(0, 20, num_samples)
        if use_bias:
            X = jnp.c_[jnp.ones(num_samples), X]
        return X
    return make_evenly_spaced_x


def sample_fn(apply_fn: Callable,
                         x_generator: Callable,
                         num_train: int,
                         key: chex.PRNGKey):
                
  """Generates training data for given problem."""
  # Checking the dimensionality of our data coming in.
  x_train = x_generator(key, num_train)

  # Generate environment function across x_train
  # Data from original matlab example
  y_train = jnp.array([2.4865, -0.3033, -4.0531, -4.3359, -6.1742, -5.604, -3.5069, -2.3257, -4.6377,
                        -0.2327, -1.9858, 1.0284, -2.264, -0.4508, 1.1672, 6.6524, 4.1452, 5.2677, 6.3403, 9.6264, 14.7842])

  data = (x_train, y_train)

  return data


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.nsteps = 21
    config.ntrials = 1

    config.save_params = True

    problem_type = "regression"
    model = "linear"

    input_dim = 2
    tau = 1
    output_dim = 1
    prior_knowledge = PriorKnowledge(input_dim, tau, output_dim=output_dim)

    # Random seed controlling all the randomness in the problem.
    config.seed = 0

    config.env = EnvironmentConfig(problem_type, model, prior_knowledge,
                                train_distribution=make_evenly_spaced_x_sampler,
                                test_distribution=make_evenly_spaced_x_sampler,
                                sample_fn=sample_fn, train_batch_size=1)

    model = "KalmanFilter"
    
    init_kwargs = dict(mu0=jnp.zeros(input_dim),
                        Sigma0=jnp.eye(input_dim) * 10.,
                        F=jnp.eye(input_dim),
                        Q=0,
                        R=1)

    config.agent = AgentConfig(model, init_kwargs=init_kwargs)

    return config
