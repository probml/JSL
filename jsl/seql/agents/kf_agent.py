# Kalman filter agent

import jax.numpy as jnp
from jax import vmap

import chex
from typing import NamedTuple

from jsl.seql.agents.base import Agent
from jsl.seql.utils import posterior_noise
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

        A, Q = jnp.eye(input_dim), 0
        C = lambda t: x[t][None, ...]

        lds = LDS(A, C, Q, obs_noise, belief.mu, belief.Sigma)
        mu, Sigma, _, _ = kalman_filter(lds, y,
                                        return_history=return_history)
        if return_history:
            history = (mu, Sigma)
            mu, Sigma = mu[-1], Sigma[-1]
            return BeliefState(mu, Sigma), Info(*history)
        
        return BeliefState(mu, Sigma), Info()

    def predict(belief: BeliefState,
                x: chex.Array):
        v_posterior_noise = vmap(posterior_noise, in_axes=(0, None, None))
        noise = v_posterior_noise(x, belief.Sigma, obs_noise)
        noise = jnp.diag(jnp.squeeze(noise))
        
        return x @ belief.mu, noise

    return Agent(init_state, update, predict)
