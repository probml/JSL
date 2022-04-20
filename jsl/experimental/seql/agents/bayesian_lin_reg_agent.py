from jax import config
from jsl.experimental.seql.agents.agent_utils import Memory

config.update('jax_default_matmul_precision', 'float32')

import jax.numpy as jnp

import distrax

import chex
from typing import NamedTuple, Tuple

# Local imports
from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.agents.kf_agent import BeliefState


class Info(NamedTuple):
    ...


class BayesianReg(Agent):

    def __init__(self,
                 buffer_size: int,
                 obs_noise: float,
                 is_classifier: bool = False):
        assert is_classifier == False
        self.memory = Memory(buffer_size)
        super(BayesianReg, self).__init__(is_classifier)

        self.buffer_size = buffer_size
        self.obs_noise = obs_noise
        self.model_fn = lambda params, x: x @ params

    def init_state(self,
                   mu: chex.Array,
                   Sigma: chex.Array) -> BeliefState:
        return BeliefState(mu, Sigma)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array) -> Tuple[BeliefState, Info]:

        assert self.buffer_size >= len(x)

        x_, y_ = self.memory.update(x, y)
        Sigma0_inv = jnp.linalg.inv(belief.Sigma)
        Sigma_inv = Sigma0_inv + (x_.T @ x_) / self.obs_noise

        Sigma = jnp.linalg.inv(Sigma_inv)
        mu = Sigma @ (Sigma0_inv @ belief.mu + x_.T @ y_ / self.obs_noise)

        return BeliefState(mu, Sigma), Info()

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState)-> chex.ArrayTree:
        mu, Sigma = belief.mu, belief.Sigma
        mvn = distrax.MultivariateNormalFullCovariance(jnp.squeeze(mu, axis=-1),
                                                       Sigma)
        theta = mvn.sample(seed=key)
        theta = theta.reshape(mu.shape)
        return theta
