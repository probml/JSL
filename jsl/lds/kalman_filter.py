# Jax implementation of a Linear Dynamical System
# Author:  Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)


from jax import config
config.update('jax_default_matmul_precision', 'float32')

import chex

import jax.numpy as jnp
from jax.random import multivariate_normal, split
from jax.numpy.linalg import inv
from jax import tree_map

from jax import lax, vmap

from dataclasses import dataclass
from functools import partial
from typing import Union, Callable

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

@dataclass
class LDS:
    """
    Implementation of the Kalman Filtering and Smoothing
    procedure of a Linear Dynamical System with known parameters.
    This class exemplifies the use of Kalman Filtering assuming
    the model parameters are known.
    Parameters
    ----------
    A: array(state_size, state_size)
        Transition matrix
    C: array(observation_size, state_size)
        Constant observation matrix or function that depends on time
    Q: array(state_size, state_size)
        Transition covariance matrix
    R: array(observation_size, observation_size)
        Observation covariance
    mu: array(state_size)
        Mean of initial configuration
    Sigma: array(state_size, state_size) or 0
        Covariance of initial configuration. If value is set
        to zero, the initial state will be completely determined
        by mu0
    """
    A: chex.Array
    C: Union[chex.Array, Callable]
    Q: chex.Array
    R: chex.Array
    mu: chex.Array
    Sigma: chex.Array

    def observations(self, t: int):
        if callable(self.C):
            return self.C(t)
        else:
            return self.C

    def sample(self,
               key: chex.PRNGKey,
               timesteps: int,
               n_samples: int=1,
               sample_intial_state: bool=False):
        """
        Simulate a run of n_sample independent stochastic
        linear dynamical systems
        Parameters
        ----------
        key: jax.random.PRNGKey
            Seed of initial random states
        timesteps: int
            Total number of steps to sample
        n_samples: int
            Number of independent linear systems with shared dynamics (optional)
        sample_initial_state: bool
            Whether to sample from an initial state or sepecified
        Returns
        -------
        * array(n_samples, timesteps, state_size):
            Simulation of Latent states
        * array(n_samples, timesteps, observation_size):
            Simulation of observed states
        """
        key_z1, key_system_noise, key_obs_noise = split(key, 3)
        state_size, _ = self.A.shape

        if not sample_intial_state:
            state_t = self.mu * jnp.ones((n_samples, state_size))
        else:
            state_t = multivariate_normal(key_z1, self.mu, self.Sigma, (n_samples,))

        # Generate all future noise terms
        zeros_state = jnp.zeros(state_size)
        observation_size = timesteps if isinstance(self.R, int) else self.R.shape[0]
        zeros_obs = jnp.zeros(observation_size)

        system_noise = multivariate_normal(key_system_noise, zeros_state, self.Q, (timesteps, n_samples))
        obs_noise = multivariate_normal(key_obs_noise, zeros_obs, self.R, (timesteps, n_samples))

        obs_t = jnp.einsum("ij,sj->si", self.observations(0), state_t) + obs_noise[0]

        def sample_step(state, carry):
            system_noise_t, obs_noise_t, t = carry
            state_new = jnp.einsum("ij,sj->si", self.A, state) + system_noise_t
            obs_new = jnp.einsum("ij,sj->si", self.observations(t), state_new) + obs_noise_t
            return state_new, (state_new, obs_new)

        timesteps = jnp.arange(1, timesteps)
        carry = (system_noise[1:], obs_noise[1:], timesteps)
        _, (state_hist, obs_hist) = lax.scan(sample_step, state_t, carry)

        state_hist = jnp.swapaxes(jnp.vstack([state_t[None, ...], state_hist]), 0, 1)
        obs_hist = jnp.swapaxes(jnp.vstack([obs_t[None, ...], obs_hist]), 0, 1)

        if n_samples == 1:
            state_hist = state_hist[0, ...]
            obs_hist = obs_hist[0, ...]
        return state_hist, obs_hist


def kalman_smoother(params: LDS,
                    mu_hist: chex.Array,
                    Sigma_hist: chex.Array,
                    mu_cond_hist: chex.Array,
                    Sigma_cond_hist: chex.Array):
    """
    Compute the offline version of the Kalman-Filter, i.e,
    the kalman smoother for the hidden state.
    Note that we require to independently run the kalman_filter function first
    Parameters
    ----------
    params: LDS
         Linear Dynamical System object
    mu_hist: array(timesteps, state_size):
        Filtered means mut
    Sigma_hist: array(timesteps, state_size, state_size)
        Filtered covariances Sigmat
    mu_cond_hist: array(timesteps, state_size)
        Filtered conditional means mut|t-1
    Sigma_cond_hist: array(timesteps, state_size, state_size)
        Filtered conditional covariances Sigmat|t-1
    Returns
    -------
    * array(timesteps, state_size):
        Smoothed means mut
    * array(timesteps, state_size, state_size)
        Smoothed covariances Sigmat
    """
    A = params.A

    mut_giv_T = mu_hist[-1, :]
    Sigmat_giv_T = Sigma_hist[-1, :]

    def smoother_step(state, elements):
        mut_giv_T, Sigmat_giv_T = state
        mutt, Sigmatt, mut_cond_next, Sigmat_cond_next = elements
        Jt = Sigmatt @ A.T @ inv(Sigmat_cond_next)
        mut_giv_T = mutt + Jt @ (mut_giv_T - mut_cond_next)
        Sigmat_giv_T = Sigmatt + Jt @ (Sigmat_giv_T - Sigmat_cond_next) @ Jt.T
        return (mut_giv_T, Sigmat_giv_T), (mut_giv_T, Sigmat_giv_T)

    elements = (mu_hist[-2::-1],
                Sigma_hist[-2::-1, ...],
                mu_cond_hist[1:][::-1, ...],
                Sigma_cond_hist[1:][::-1, ...])
    initial_state = (mut_giv_T, Sigmat_giv_T)

    _, (mu_hist_smooth, Sigma_hist_smooth) = lax.scan(smoother_step, initial_state, elements)

    mu_hist_smooth = jnp.concatenate([mu_hist_smooth[::-1, ...], mut_giv_T[None, ...]], axis=0)
    Sigma_hist_smooth = jnp.concatenate([Sigma_hist_smooth[::-1, ...], Sigmat_giv_T[None, ...]], axis=0)

    return mu_hist_smooth, Sigma_hist_smooth

        

def kalman_filter(params: LDS, x_hist: chex.Array,
                  return_history: bool = True):
    """
    Compute the online version of the Kalman-Filter, i.e,
    the one-step-ahead prediction for the hidden state or the
    time update step
    Parameters
    ----------
    params: LDS
         Linear Dynamical System object
    x_hist: array(timesteps, observation_size)
    return_history: bool
    Returns
    -------
    * array(timesteps, state_size):
        Filtered means mut
    * array(timesteps, state_size, state_size)
        Filtered covariances Sigmat
    * array(timesteps, state_size)
        Filtered conditional means mut|t-1
    * array(timesteps, state_size, state_size)
        Filtered conditional covariances Sigmat|t-1
    """
    A, Q, R = params.A, params.Q, params.R

    state_size, _ = A.shape
    I = jnp.eye(state_size)


    def predict_step(mu, Sigma):
        # \Sigma_{t|t-1}
        Sigman_cond = A @ Sigma @ A.T + Q

        # \mu_{t |t-1} and xn|{n-1}
        mu_cond = A @ mu
        
        return mu_cond, Sigman_cond

    def kalman_step(state, obs):
        mu, Sigma, t = state
        
        mu_cond, Sigma_cond = predict_step(mu, Sigma)
        Ct = params.observations(t)

        St = Ct @ Sigma_cond @ Ct.T + R
        Kt = Sigma_cond @ Ct.T @ inv(St)
        
        et = obs - Ct @ mu_cond
        mu = mu_cond + Kt @ et

        #  More stable solution is (I − KtCt)Σt|t−1(I − KtCt)T + KtRtKTt
        tmp = (I - Kt @ Ct)
        Sigma = tmp @ Sigma_cond @ tmp.T +  Kt @ (R * Kt.T)

        t = t + 1

        return (mu, Sigma, t), (mu, Sigma, mu_cond, Sigma_cond)

    mu0, Sigma0 = params.mu, params.Sigma
    initial_state = (mu0, Sigma0, 0)
    (mun, Sigman, _), history = lax.scan(kalman_step, initial_state, x_hist)
    if return_history:
        return history
    return mun, Sigman, None, None


def filter(params: LDS, x_hist: chex.Array,
           return_history: bool = True):
    """
    Compute the online version of the Kalman-Filter, i.e,
    the one-step-ahead prediction for the hidden state or the
    time update step.
    Note that x_hist can optionally be of dimensionality two,
    This corresponds to different samples of the same underlying
    Linear Dynamical System
    Parameters
    ----------
    params: LDS
         Linear Dynamical System object
    x_hist: array(n_samples?, timesteps, observation_size)
    return_history: bool
    Returns
    -------
    * array(n_samples?, timesteps, state_size):
        Filtered means mut
    * array(n_samples?, timesteps, state_size, state_size)
        Filtered covariances Sigmat
    * array(n_samples?, timesteps, state_size)
        Filtered conditional means mut|t-1
    * array(n_samples?, timesteps, state_size, state_size)
        Filtered conditional covariances Sigmat|t-1
    """
    has_one_sim = False
    if x_hist.ndim == 2:
        x_hist = x_hist[None, ...]
        has_one_sim = True

    kalman_map = vmap(partial(kalman_filter, return_history=return_history), (None, 0))
    outputs = kalman_map(params, x_hist)

    if has_one_sim and return_history:
        outputs = tree_map(lambda x: x[0, ...], outputs)
        return outputs

    return outputs


def smooth(params: LDS,
           mu_hist: chex.Array,
           Sigma_hist: chex.Array,
           mu_cond_hist: chex.Array,
           Sigma_cond_hist: chex.Array):
    """
    Compute the offline version of the Kalman-Filter, i.e,
    the kalman smoother for the state space.
    Note that we require to independently run the kalman_filter function first.
    Note that the mean terms can optionally be of dimensionality two.
    Similarly, the covariance terms can optinally be of dimensionally three.
    This corresponds to different samples of the same underlying
    Linear Dynamical System
    Parameters
    ----------
    params: LDS
         Linear Dynamical System object
    mu_hist: array(n_samples?, timesteps, state_size):
        Filtered means mut
    Sigma_hist: array(n_samples?, timesteps, state_size, state_size)
        Filtered covariances Sigmat
    mu_cond_hist: array(n_samples?, timesteps, state_size)
        Filtered conditional means mut|t-1
    Sigma_cond_hist: array(n_samples?, timesteps, state_size, state_size)
        Filtered conditional covariances Sigmat|t-1
    Returns
    -------
    * array(n_samples?, timesteps, state_size):
        Smoothed means mut
    * array(timesteps?, state_size, state_size)
        Smoothed covariances Sigmat
    """
    has_one_sim = False
    if mu_hist.ndim == 2:
        mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = mu_hist[None, ...], Sigma_hist[None, ...], \
                                                             mu_cond_hist[None, ...], Sigma_cond_hist[None, ...]
        has_one_sim = True
    smoother_map = vmap(kalman_smoother, (None, 0, 0, 0, 0))
    mu_hist_smooth, Sigma_hist_smooth = smoother_map(params, mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist)
    if has_one_sim:
        mu_hist_smooth, Sigma_hist_smooth = mu_hist_smooth[0, ...], Sigma_hist_smooth[0, ...]
    return mu_hist_smooth, Sigma_hist_smooth