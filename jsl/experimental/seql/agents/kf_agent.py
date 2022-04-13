# Kalman filter agent
from jax import config

config.update('jax_default_matmul_precision', 'float32')

import jax.numpy as jnp

import distrax

import chex
from typing import NamedTuple

from jsl.experimental.seql.agents.base import Agent
from jsl.lds.kalman_filter import LDS, kalman_filter


class BeliefState(NamedTuple):
    mu: chex.Array
    Sigma: chex.Array


class Info(NamedTuple):
    mu_hist: chex.Array = None
    Sigma_hist: chex.Array = None


def kalman_filter_reg(obs_noise: float = 1.,
                      return_history: bool = False):
    classification = False

    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        *_, input_dim = x.shape

        A, Q = jnp.eye(input_dim), 0
        C = lambda t: x[t][None, ...]

        lds = LDS(A, C, Q, obs_noise, belief.mu, belief.Sigma)
        mu, Sigma, _, _ = kalman_filter(lds, y,
                                        return_history=return_history)
        if return_history:
            history = (mu, Sigma)
            mu, Sigma = mu[-1], Sigma[-1]
            return BeliefState(mu, Sigma), Info(*history)

        return BeliefState(mu.reshape((-1, 1)), Sigma), Info()

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
