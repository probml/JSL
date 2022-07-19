# Jax implementation of a Linear Dynamical System
# Author:  Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna), Kevin Murphy (@murphyk)

import chex
import jax.numpy as jnp
from jax.random import multivariate_normal, split
from jax.scipy.linalg import solve
from jax import tree_map, lax, vmap
from dataclasses import dataclass, field
from functools import partial
from typing import Union, Callable

ArrayOrFn = Union[chex.Array, Callable]

@dataclass
class LDS:
    """
    Implementation of the Kalman Filtering and Smoothing
    procedure of a Linear Dynamical System (LDS) with known parameters.
    This class exemplifies the use of Kalman Filtering assuming
    the model parameters are known.

    The LDS evolves as follows:
    x_t = A x_t-1 + w_t; w_t ~ N(state_offset, Q)
    y_t = C x_t + v_t; v_t ~ N(obs_offset, R)

    with initial state x_0 ~ N(mu, Sigma)

    Parameters
    ----------
    A: array(state_size, state_size)
        Transition matrix or function that depends on time
    C: array(observation_size, state_size)
        Constant observation matrix or function that depends on time
    Q: array(state_size, state_size)
        Transition covariance matrix or function that depends on time
    R: array(observation_size, observation_size)
        Observation covariance or function that depends on time
    mu: array(state_size)
        Mean of initial configuration
    Sigma: array(state_size, state_size) or 0
        Covariance of initial configuration. If value is set
        to zero, the initial state will be completely determined
        by mu0
    
    """
    A: ArrayOrFn
    C: ArrayOrFn
    Q: ArrayOrFn
    R: ArrayOrFn
    mu: chex.Array
    Sigma: chex.Array

    state_offset: ArrayOrFn = None
    obs_offset: ArrayOrFn = None

    nstates: int = field(init=False)
    nobs: int = field(init=False)


    def get_trans_mat_of(self, t: int):
        if callable(self.A):
            return self.A(t)
        else:            
            return self.A

    def get_obs_mat_of(self, t: int):
        if callable(self.C):
            return self.C(t)
        else:
            return self.C
        
    def get_system_noise_of(self, t: int):
        if callable(self.Q):
            return self.Q(t)
        else:
            return self.Q

    def get_observation_noise_of(self, t: int):
        if callable(self.R):
            return self.R(t)
        else:
            return self.R

    def get_state_offset_of(self, t: int):
        if self.state_offset is None:
          return jnp.zeros((self.nstates))
        elif callable(self.state_offset):
            return self.state_offset(t)
        else:
            return self.state_offset

    def get_obs_offset_of(self, t: int):
        if self.obs_offset is None:
            return jnp.zeros((self.nobs))
        elif callable(self.obs_offset):
            return self.obs_offset(t)
        else:
            return self.obs_offset

    def __post_init__(self):
            self.nobs, self.nstates = self.get_obs_mat_of(0).shape


    def sample(self,
               key: chex.PRNGKey,
               timesteps: int,
               n_samples: int = 1,
               sample_initial_state: bool = False):
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
            Whether to sample from an initial state or specified
        Returns
        -------
        * array(n_samples, timesteps, state_size):
            Simulation of Latent states
        * array(n_samples, timesteps, observation_size):
            Simulation of observed states
        """
        key_z1, key_system_noise, key_obs_noise = split(key, 3)
        state_size, _ = self.get_trans_mat_of(0).shape

        if not sample_initial_state:
            state_t = self.mu * jnp.ones((n_samples, state_size))
        else:
            state_t = multivariate_normal(key_z1, self.mu, self.Sigma, (n_samples,))

        # Generate all future noise terms
        zeros_state = jnp.zeros(state_size)
        Q = self.get_system_noise_of(0) # assumed static
        R = self.get_observation_noise_of(0) # assumed static
  

        observation_size = timesteps if isinstance(R, int) else R.shape[0]
        zeros_obs = jnp.zeros(observation_size)

        system_noise = multivariate_normal(key_system_noise, zeros_state, Q, (timesteps, n_samples))
        obs_noise = multivariate_normal(key_obs_noise, zeros_obs, R, (timesteps, n_samples))

        # observation at time t=0
        obs_t = jnp.einsum("ij,sj->si", self.get_obs_mat_of(0), state_t) + obs_noise[0]

        def sample_step(state, inps):
            system_noise_t, obs_noise_t, t = inps
            A = self.get_trans_mat_of(t)
            C = self.get_obs_mat_of(t)
            state_new = state @ A.T + system_noise_t
            state_new = state_new + self.get_state_offset_of(t)
            obs_new = state_new @ C.T + obs_noise_t
            obs_new = obs_new + self.get_obs_offset_of(t)
            return state_new, (state_new, obs_new)

        timesteps = jnp.arange(1, timesteps)
        inputs = (system_noise[1:], obs_noise[1:], timesteps)
        _, (state_hist, obs_hist) = lax.scan(sample_step, state_t, inputs)

        state_hist = jnp.swapaxes(jnp.vstack([state_t[None, ...], state_hist]), 0, 1)
        obs_hist = jnp.swapaxes(jnp.vstack([obs_t[None, ...], obs_hist]), 0, 1)

        if n_samples == 1:
            state_hist = state_hist[0, ...]
            obs_hist = obs_hist[0, ...]
        return state_hist, obs_hist


def kalman_predict(mu, Sigma, t, params):
    """
    Calculate the Kalman filter predict step.

    Parameters
    ----------
    mu: array(state_size)
         Posterior estimate of mu from previous time step, mu_{t-1}.
    Sigma: array(state_size, state_size)
         Posterior estimate of Sigma from previous time step, Sigma_{t-1}.
    t: int
    params: LDS
         Linear Dynamical System object containing parameters.

    Returns
    -------
    * array(state_size):
        Predicted mean at current time step, mu_{t|t-1}
    * array(state_size, state_size)
        Predicted covariance at current time step, Sigma_{t|t-1}
    """

    # \Sigma_{t|t-1}
    A = params.get_trans_mat_of(t)
    Q = params.get_system_noise_of(t)

    Sigma_pred = A @ Sigma @ A.T + Q

    # \mu_{t |t-1} and xn|{n-1}
    mu_pred = A @ mu
    mu_pred = mu_pred + params.get_state_offset_of(t)
    return mu_pred, Sigma_pred


def kalman_update(mu_pred, Sigma_pred, obs, t, params):
    """
    Calculate the Kalman filter update step.

    Parameters
    ----------
    mu_pred: array(state_size)
         Prediction of mu_{t|t-1} from mu_{t-1} and dynamics.
    Sigma_pred: array(state_size, state_size)
         Prediction of Sigma_{t|t-1} from Sigma_{t-1} and dynamics.
    obs: array(observation_size)
         Observation at time, t.
    t: int
    params: LDS
         Linear Dynamical System object containing parameters.

    Returns
    -------
    * array(state_size):
        Updated mean mu_t
    * array(state_size, state_size)
        Updated covariance Sigma_t
    """

    Ct = params.get_obs_mat_of(t)
    R = params.get_observation_noise_of(t)

    St = Ct @ Sigma_pred @ Ct.T + R
    Kt = solve(St, Ct @ Sigma_pred, sym_pos=True).T

    pred_obs = Ct @ mu_pred
    pred_obs = pred_obs + params.get_obs_offset_of(t)
    innovation = obs - pred_obs
    mu = mu_pred + Kt @ innovation

    #  More stable solution is (I − KtCt)Σt|t−1(I − KtCt)T + KtRtKTt
    I = jnp.eye(len(params.mu))
    tmp = I - Kt @ Ct
    Sigma = tmp @ Sigma_pred @ tmp.T + Kt @ (R @ Kt.T)

    return mu, Sigma


def kalman_step(state, obs, params):
    """
    A single step of the Kalman filter to calculate mu_t, Sigma_t given estimates 
     at previous time step and observation and system parameters at time t.

    Parameters
    ----------
    state: tuple(mu, Sigma, t).
        mu: array(state_size).
            Posterior mu from previous time step, mu_{t-1}.
        Sigma: array(state_size, state_size)
            Posterior estimate of Sigma from previous time step, Sigma_{t-1}.
        t: int
    obs: array(observation_size)
         Observation at time, t.
    params: LDS
         Linear Dynamical System object containing parameters.

    Returns
    -------
    * tuple(mu, Sigma, t)
    * tuple(mu, Sigma, mu_pred, Sigma_pred)
        mu: array(state_size).
            Posterior mu from current time step, mu_t.
        Sigma: array(state_size, state_size)
            Posterior Sigma from current time step, Sigma_t.
        t: int.
        mu_pred: array(state_size).
            Predicted mean mu_{t|t-1}
        Sigma_pred: array(state_size, state_size).
            Predicted covariance Sigma_{t|t-1}
    """
    mu, Sigma, t = state
    mu_pred, Sigma_pred = kalman_predict(mu, Sigma, t, params)
    mu, Sigma = kalman_update(mu_pred, Sigma_pred, obs, t, params)

    return (mu, Sigma, t + 1), (mu, Sigma, mu_pred, Sigma_pred)


def kalman_filter(params: LDS, x_hist: chex.Array, return_history: bool = True):
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
        Filtered means mu_{1:T}
    * array(timesteps, state_size, state_size)
        Filtered covariances Sigma_{1:T}
    * array(timesteps, state_size)
        Filtered conditional means mu_{t|t-1} for t=1, ..., T.
    * array(timesteps, state_size, state_size)
        Filtered conditional covariances Sigma_{t|t-1} for t=1, ..., T.
    """
    mu0, Sigma0 = params.mu, params.Sigma
    x0 = x_hist[0, ...]
    initial_state = (mu0, Sigma0, 0)
    # Filtering the first observation does not include a predict step.
    mu0_post, Sigma0_post = kalman_update(mu0, Sigma0, x0, 0, params)

    kalman_step_run = partial(kalman_step, params=params)
    (mu_T, Sigma_T, _), history = lax.scan(
        kalman_step_run, (mu0_post, Sigma0_post, 1), x_hist[1:, ...]
    )
    if return_history:
        mu_hist, Sigma_hist, *pred_hist = history
        # Add the initial update step to the history
        mu_hist = jnp.vstack((mu0_post, mu_hist))
        Sigma_hist = jnp.concatenate((Sigma0_post[None, ...], Sigma_hist))
        return (mu_hist, Sigma_hist, *pred_hist)
    return mu_T, Sigma_T, None, None


def filter(params: LDS,
           x_hist: chex.Array,
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


def smoother_step(state, elements, params):
    mut_giv_T, Sigmat_giv_T, t = state
    A = params.get_trans_mat_of(t)

    mutt, Sigmatt, mut_cond_next, Sigmat_cond_next = elements

    Jt = solve(Sigmat_cond_next, A @ Sigmatt, sym_pos=True).T
    mut_giv_T = mutt + Jt @ (mut_giv_T - mut_cond_next)
    Sigmat_giv_T = Sigmatt + Jt @ (Sigmat_giv_T - Sigmat_cond_next) @ Jt.T
    return (mut_giv_T, Sigmat_giv_T,  t+1), (mut_giv_T, Sigmat_giv_T)


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

    mut_giv_T = mu_hist[-1, :]
    Sigmat_giv_T = Sigma_hist[-1, :]

    smoother_step_run = partial(smoother_step, params=params)
    elements = (mu_hist[-2::-1],
                Sigma_hist[-2::-1, ...],
                mu_cond_hist[::-1, ...],
                Sigma_cond_hist[::-1, ...])
    initial_state = (mut_giv_T, Sigmat_giv_T, 0)

    _, (mu_hist_smooth, Sigma_hist_smooth) = lax.scan(smoother_step_run, initial_state, elements)

    mu_hist_smooth = jnp.concatenate([mu_hist_smooth[::-1, ...], mut_giv_T[None, ...]], axis=0)
    Sigma_hist_smooth = jnp.concatenate([Sigma_hist_smooth[::-1, ...], Sigmat_giv_T[None, ...]], axis=0)

    return mu_hist_smooth, Sigma_hist_smooth


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
    mu_hist_smooth, Sigma_hist_smooth = smoother_map(params, mu_hist, Sigma_hist,
                                                     mu_cond_hist, Sigma_cond_hist)
    if has_one_sim:
        mu_hist_smooth, Sigma_hist_smooth = mu_hist_smooth[0, ...], Sigma_hist_smooth[0, ...]
    return mu_hist_smooth, Sigma_hist_smooth
