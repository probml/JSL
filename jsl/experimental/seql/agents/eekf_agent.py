# EEKF agent
import jax.numpy as jnp

import chex
import distrax

from typing import List

from jsl.experimental.seql.agents.base import Agent
from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter
from jsl.experimental.seql.agents.kf_agent import BeliefState, Info


class EEKFAgent(Agent):

    def __init__(self,
                 nlds: NLDS,
                 return_params: List[str] = ["mean", "cov"],
                 return_history: bool = False,
                 is_classifier: bool = True):
        assert is_classifier == True
        super(EEKFAgent, self).__init__(is_classifier)

        self.nlds = nlds
        self.return_params = return_params
        self.return_history = return_history
        self.model_fn = lambda params, x: x @ params

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

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):
        n = len(x)
        posterior_cov = x @ belief.Sigma @ x.T + self.obs_noise * jnp.eye(n)
        chex.assert_shape(posterior_cov, [n, n])
        return posterior_cov

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        mu, Sigma = belief.mu, belief.Sigma
        mvn = distrax.MultivariateNormalFullCovariance(mu, Sigma)
        theta = mvn.sample(seed=key, sample_shape=mu.shape)
        return theta
