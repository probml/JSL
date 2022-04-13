from jax import config, random, vmap

from jsl.experimental.seql.agents.agent_utils import Memory

config.update('jax_default_matmul_precision', 'float32')

import jax.numpy as jnp

import distrax

import chex
from typing import NamedTuple

# Local imports
from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.agents.kf_agent import BeliefState


class Info(NamedTuple):
    ...


def bayesian_reg(buffer_size: int,
                 obs_noise: float = 1.):
    classification = False
    memory = Memory(buffer_size)

    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert buffer_size >= len(x)

        x_, y_ = memory.update(x, y)
        Sigma0_inv = jnp.linalg.inv(belief.Sigma)
        Sigma_inv = Sigma0_inv + (x_.T @ x_) / obs_noise

        Sigma = jnp.linalg.inv(Sigma_inv)
        mu = Sigma @ (Sigma0_inv @ belief.mu + x_.T @ y_ / obs_noise)

        return BeliefState(mu, Sigma), Info()

    def apply(params: chex.ArrayTree,
              x: chex.Array):
        n = len(x)
        predictions = x @ params
        predictions = predictions.reshape((n, -1))

        return predictions

    def sample_params(key: chex.PRNGKey,
                      belief: BeliefState):
        mu, Sigma = belief.mu, belief.Sigma
        mvn = distrax.MultivariateNormalFullCovariance(mu, Sigma)
        theta = mvn.sample(seed=key, sample_shape=mu.shape)

        return theta

    return Agent(classification, init_state, update, apply, sample_params)
