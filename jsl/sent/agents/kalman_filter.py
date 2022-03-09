# Kalman filter agent

import chex

import jax.numpy as jnp
from jax.lax import scan

import haiku as hk

from jsl.lds.kalman_filter import KalmanFilter
from jsl.sent.agents.agent import Agent

class KalmanFilterReg(Agent):

    def __init__(self,
                mu0: chex.Array,
                Sigma0:chex.Array,
                F: chex.Array,
                Q: float, R: float):
        
        self.rng = hk.PRNGSequence(0)

        # Prior mean
        self.prior_mean = mu0
        self.mu = mu0

        # Prior covariance matrix
        self.prior_sigma= Sigma0
        self.Sigma = Sigma0

        self.F = F
        # Known variance
        self.Q = Q
        self.R = R
        
        self.mu_hist = None
        self.Sigma_hist = None

    def update(self,
                X: chex.Array, y: chex.Array):
        ntrain, unused_dim = X.shape
        chex.assert_shape(y, [ntrain, 1])
        C = lambda t: X[t][None, ...]
        kf = KalmanFilter(self.F, C, self.Q, self.R, self.mu, self.Sigma, timesteps=ntrain)
        (mu, Sigma, _), (mu_hist, Sigma_hist, *_) = scan(kf.kalman_step, (self.mu, self.Sigma, 0), y)
        self.mu, self.Sigma = mu, Sigma
        return {"mean": mu_hist, "cov": Sigma_hist}

    def predict(self, x: chex.Array):
        return x @ self.mu

    def reset(self, key: chex.Array):
        self.mu = self.prior_mean
        self.Sigma = self.prior_sigma