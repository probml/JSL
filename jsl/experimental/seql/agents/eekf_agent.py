# EEKF agent
import jax.numpy as jnp

import chex
import distrax

from typing import List, Callable

from jsl.experimental.seql.agents.base import Agent
from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter
from jsl.experimental.seql.agents.kf_agent import BeliefState, Info


class EEKFAgent(Agent):

    def __init__(self,
                 nlds: NLDS,
                 model_fn: Callable = lambda params, x: x @ params,
                 obs_noise: float = 0.1,
                 return_params: List[str] = ["mean", "cov"],
                 return_history: bool = False,
                 is_classifier: bool = True):
        assert is_classifier == True
        super(EEKFAgent, self).__init__(is_classifier)

        self.nlds = nlds
        self.return_params = return_params
        self.return_history = return_history
        self.model_fn = model_fn
        self.obs_noise = obs_noise

    def init_state(self,
                   mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        (mu, Sigma), history = filter(self.nlds,
                                      belief.mu,
                                      y, x, belief.Sigma,
                                      self.return_params,
                                      return_history=self.return_history)
        if self.return_history:
            return BeliefState(mu, Sigma), Info(history["mean"], history["cov"])

        return BeliefState(mu, Sigma), Info()

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        mu, Sigma = belief.mu, belief.Sigma
        print(mu.shape, Sigma.shape)
        mvn = distrax.MultivariateNormalFullCovariance(jnp.squeeze(mu, axis=-1),
                                                       Sigma)
        theta = mvn.sample(seed=key)
        theta = theta.reshape(mu.shape)
        return theta
