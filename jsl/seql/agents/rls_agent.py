# Kalman filter agent

from jax import config
config.update('jax_default_matmul_precision', 'float32')

import jax.numpy as jnp
from jax import vmap, lax

import chex
from typing import NamedTuple

from jsl.seql.agents.base import Agent
from jsl.seql.utils import posterior_noise

class BeliefState(NamedTuple):
    mu: chex.Array
    Sigma: chex.Array


class Info(NamedTuple):
    mu_hist: chex.Array = None
    Sigma_hist: chex.Array = None


def rls_agent(obs_noise: float = 1., return_history=True):
    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
    
        def step(state, carry):
            mu, Sigma = state
            xt_, yt = carry
            xt = xt_.reshape((-1, 1))

            # st = xTt Σt−1 xt + σ2
            st = jnp.matmul(xt.T, jnp.matmul(Sigma, xt)) + obs_noise

            #μt = μt−1 + Σt−1xt(yt − xTt μt−1)
            mut = mu + jnp.matmul(Sigma,
                                  jnp.matmul(xt,
                                            yt - jnp.matmul(xt.T,mu)
                                            )
                                  ) / st
            Sigmat = Sigma - jnp.matmul(Sigma, jnp.matmul(xt,jnp.matmul(xt.T, Sigma))) / st

            return (mut, Sigmat), (mut, Sigmat)
        
        (mu, Sigma), history = lax.scan(step, (belief.mu, belief.Sigma), (x, y))
        
        if return_history:
            BeliefState(mu, Sigma), Info(*history)

        return BeliefState(mu, Sigma), Info()

    def predict(belief: BeliefState,
                x: chex.Array):
        v_posterior_noise = vmap(posterior_noise, in_axes=(0, None, None))
        noise = v_posterior_noise(x, belief.Sigma, obs_noise)
        noise = jnp.diag(jnp.squeeze(noise))
        
        return x @ belief.mu, noise

    return Agent(init_state, update, predict)
