# EEKF agent
from jax import random, vmap

import chex
import distrax

from typing import List

from jsl.experimental.seql.agents.base import Agent
from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter
from jsl.experimental.seql.agents.kf_agent import BeliefState, Info


def eekf(nlds: NLDS,
         return_params: List[str] = ["mean", "cov"],
         return_history: bool = False):
    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        (mu, Sigma), history = filter(nlds, belief.mu,
                                      y, x, belief.Sigma,
                                      return_params,
                                      return_history=return_history)
        if return_history:
            return BeliefState(mu, Sigma), Info(history["mean"], history["cov"])

        return BeliefState(mu, Sigma), Info()

    def predict(belief: BeliefState,
                x: chex.Array):
        nsamples = len(x)
        predictions = x @ belief.mu
        predictions = predictions.reshape((nsamples, -1))

        return predictions

    def sample_predictive(key: chex.PRNGKey,
                             belief: BeliefState,
                             x: chex.Array,
                             nsamples: int):
        keys = random.split(key, nsamples)
        mvn = distrax.MultivariateNormalFullCovariance(belief.mu,
                                                       belief.Sigma)

        def predict(key: chex.PRNGKey):
            params = mvn.sample(seed=key, shape=belief.mu.shape)
            return x @ params

        return vmap(predict)(keys)

    return Agent(init_state, update, predict, sample_predictive)
