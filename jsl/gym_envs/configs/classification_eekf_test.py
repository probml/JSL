import jax.numpy as jnp
import ml_collections
from jsl.gym_envs.configs.utils import AgentConfig, EnvironmentConfig, PriorKnowledge

from jax.nn import sigmoid
from typing import Callable
import chex

# Import data and baseline solution
from jsl.demos import logreg_biclusters as demo


_, data = demo.main()


def make_bicluster_data_sampler(input_dim: int):
    def bicluster_data_sampler(key: chex.PRNGKey, num_samples: int):
        return data["Phi"]
    return bicluster_data_sampler


def sample_fn(apply_fn: Callable,
                         x_generator: Callable,
                         num_train: int,
                         key: chex.PRNGKey):
  x_train = x_generator(key, num_train)
  y_train = data["y"]
  return (x_train, y_train), None, None


def fz(x): return x
def fx(w, x): return sigmoid(w[None, :] @ x)
def Rt(w, x): return (sigmoid(w @ x) * (1 - sigmoid(w @ x)))[None, None]


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    config.nsteps, input_dim = data["Phi"].shape
    config.ntrials = 1

    config.save_params = True

    problem_type = "classification"
    model = "linear"

    tau = 1
    output_dim = 1

    prior_knowledge = PriorKnowledge(input_dim, tau, output_dim=output_dim)

    # Random seed controlling all the randomness in the problem.
    config.seed = 0

    config.env = EnvironmentConfig(problem_type, model, prior_knowledge,
                                train_distribution=make_bicluster_data_sampler,
                                test_distribution=make_bicluster_data_sampler,
                                sample_fn=sample_fn, train_batch_size=1)

    model = "EEKF"

    ### EEKF Approximation
    mu_t = jnp.zeros(output_dim)
    Pt = jnp.eye(output_dim) * 0.0
    P0 = jnp.eye(output_dim) * 2.0

    init_kwargs = dict(fz=fz,
                        fx=fx,
                        Pt=Pt,
                        Rt=Rt,
                        mu=mu_t,
                        P0=P0)

    config.agent = AgentConfig(model, init_kwargs=init_kwargs)

    return config
