# EEKF agent
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
    classification = True

    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        (mu, Sigma), history = filter(nlds, belief.mu,
                                      y, x, belief.Sigma,
                                      return_params,
                                      return_history=return_history)
        if return_history:
            return BeliefState(mu, Sigma), Info(history["mean"], history["cov"])

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
