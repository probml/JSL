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
    mu_hist: chex.Array = None
    Sigma_hist: chex.Array = None


def kalman_filter_reg(obs_noise: float = 1.,
                      return_history: bool = False):
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
        mu, Sigma, _, _ = kalman_filter(lds, y,
                                        return_history=return_history)
        if return_history:
            history = (mu, Sigma)
            mu, Sigma = mu[-1], Sigma[-1]
            return BeliefState(mu, Sigma), Info(*history)
        
        return BeliefState(mu, Sigma), Info()

    def predict(belief: BeliefState,
                x: chex.Array):
        d, *_ = x.shape
        return x @ belief.mu, obs_noise * jnp.eye(d)

    return Agent(init_state, update, predict)
