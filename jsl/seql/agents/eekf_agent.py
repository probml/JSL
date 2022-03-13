# EEKF agent
import chex
from typing import List

from jsl.seql.agents.agent import Agent
from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter
from jsl.seql.agents.kf_agent import BeliefState, Info


def eekf(nlds: NLDS,
         return_params: List[str] = ["mean", "cov"]):

    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        (mu, Sigma), history = filter(nlds, belief.mu,
                                      y, x, belief.Sigma,
                                      return_params)

        return BeliefState(mu, Sigma), Info(history["mean"], history["cov"])

    def predict(belief: BeliefState,
                x: chex.Array):
        return x @ belief.mu

    return Agent(init_state, update, predict)
