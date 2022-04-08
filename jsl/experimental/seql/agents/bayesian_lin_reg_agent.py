from jax import config

from jsl.experimental.seql.agents.agent_utils import Memory

config.update('jax_default_matmul_precision', 'float32')

import jax.numpy as jnp

import chex
from typing import NamedTuple

# Local imports
from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.agents.kf_agent import BeliefState


class Info(NamedTuple):
    ...


def bayesian_reg(buffer_size: int, obs_noise: float = 1.):
    memory = Memory(buffer_size)

    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)
        Sigma0_inv = jnp.linalg.inv(belief.Sigma)
        Sigma_inv = Sigma0_inv + (x_.T @ x_) / obs_noise
        Sigma = jnp.linalg.inv(Sigma_inv)
        mu = Sigma @ (Sigma0_inv @ belief.mu + x_.T @ y_ / obs_noise)
        return BeliefState(mu, Sigma), Info()

    def predict(belief: BeliefState,
                x: chex.Array):
        nsamples = len(x)
        predictions = x @ belief.mu
        predictions = predictions.reshape((nsamples, -1))

        return predictions

    return Agent(init_state, update, predict)
