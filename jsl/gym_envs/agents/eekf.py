# Kalman filter agent

import chex
from typing import  Callable, List

from jsl.nlds.extended_kalman_filter import ExtendedKalmanFilter

class EmbeddedExtendedKalmanFilter:

    def __init__(self,
                fz: Callable,
                fx: Callable,
                Pt: chex.Array,
                Rt: Callable,
                mu: chex.Array,
                P0:chex.Array,
                return_params: List[str]=["mean", "cov"]):

        
        self.fz = fz
        self.fx = fx
        self.Pt = Pt
        self.Rt = Rt
        self.return_params = return_params
        
        self.prior_mean = mu
        self.prior_cov  = P0

        self.reset(None)
        self.Sigma = self.eekf.Q(self.mu)


    def update(self,
                X: chex.Array, y: chex.Array):
        (self.mu, self.Sigma, _), params = self.eekf.filter(self.mu, y, observations=X,
                        Vinit=self.Sigma,
                        return_params=self.return_params)
        return params

    def predict(self, x: chex.Array):
        return x @ self.mu

    def reset(self, key: chex.Array):
        self.eekf = ExtendedKalmanFilter(self.fz, self.fx, self.Pt, self.Rt)
        self.mu = self.prior_mean
        self.Sigma = self.prior_cov