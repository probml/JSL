# Kalman filter agent

import jax.numpy as jnp

import chex
from typing import NamedTuple

from jsl.seql.agents.agent import Agent
from jsl.lds.kalman_filter import LDS, kalman_filter


class BeliefState(NamedTuple):
    mu: chex.Array
    Sigma: chex.Array


class Info(NamedTuple):
    mu_hist: chex.Array
    Sigma_hist: chex.Array


def kalman_filter_reg(obs_noise: float = 1.):
    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        _, input_dim = x.shape

        F, Q = jnp.eye(input_dim), 0
        C = lambda t: x[t][None, ...]

        lds = LDS(F, C, Q, obs_noise, belief.mu, belief.Sigma)
        mu_hist, Sigma_hist, _, _ = kalman_filter(lds, y)

        mu, Sigma = mu_hist[-1], Sigma_hist[-1]

        return BeliefState(mu, Sigma), Info(mu_hist, Sigma_hist)

    def predict(belief: BeliefState,
                x: chex.Array):
        return x @ belief.mu

    return Agent(init_state, update, predict)
