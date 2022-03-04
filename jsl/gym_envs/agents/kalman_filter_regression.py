# Kalman filter agent
import chex
import haiku as hk

from jax.lax import scan

from .kalman_filter import KalmanFilter


class KalmanFilterReg:

    def __init__(self,
                 mu0: chex.Array,
                 Sigma0: chex.Array,
                 F: chex.Array,
                 Q: float, R: float):
        self.R = R
        # Prior mean
        self.prior_mean = mu0
        self.mu = mu0
        # Prior covariance matrix
        self.prior_sigma = Sigma0
        self.Sigma = Sigma0
        self.F = F
        # Known variance
        self.Q = Q
        self.rng = hk.PRNGSequence(0)

    def update(self,
               X: chex.Array, y: chex.Array):
        """
        Online estimation of a linear regression
        using Kalman Filters

        Parameters
        ----------
        X: array(n_obs, dimension)
            Matrix of features
        y: array(n_obs,)
            Array of observations

        Returns
        -------
        * array(n_obs, dimension)
            Online estimation of parameters
        * array(n_obs, dimension, dimension)
            Online estimation of uncertainty
        """

        def iter_fn(params, carry):
            mu, Sigma = params
            X, y = carry
            n_obs, dim = X.shape
            C = lambda t: X[t][None, ...]
            kf = KalmanFilter(self.F, C, self.Q, self.R, mu.copy(), Sigma.copy(), timesteps=n_obs)
            _, (mu_hist, Sigma_hist, _, _) = scan(kf.kalman_step, (mu.copy(), Sigma.copy(), 0), y)
            params = (mu_hist[-1], Sigma_hist[-1])
            return params, params

        if len(X.shape) == 2:
            n_obs, dim = X.shape
            chex.assert_shape(y, [n_obs, 1])
            C = lambda t: X[t][None, ...]
            kf = KalmanFilter(self.F, C, self.Q, self.R, self.mu.copy(), self.Sigma.copy(), timesteps=n_obs)
            _, (mu_hist, Sigma_hist, _, _) = scan(kf.kalman_step, (self.mu.copy(), self.Sigma.copy(), 0), y)
            mu, Sigma = mu_hist[-1], Sigma_hist[-1]
        elif len(X.shape) == 3:
            n_obs, batch_size, _ = X.shape
            chex.assert_shape(y, [n_obs, batch_size, 1])
            (mu, Sigma), (mu_hist, Sigma_hist) = scan(iter_fn, (self.mu, self.Sigma), (X, y))
        else:
            raise TypeError("The dimension of feature matrix should be greater than or equal to either 2 or 3.")

        self.mu = mu
        self.Sigma = Sigma
        self.mu_hist, self.Sigma_hist = mu_hist, Sigma_hist
        return mu_hist, Sigma_hist

    def predict(self, x):
        return x @ self.mu

    def reset(self, key):
        self.mu = self.prior_mean
        self.Sigma = self.prior_sigma
