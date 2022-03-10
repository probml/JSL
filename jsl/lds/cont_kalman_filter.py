# Implementation of the Kalman Filter for 
# continuous time series
# Author: Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)

import jax.numpy as jnp
from jax import random, lax
from jax.numpy.linalg import inv

import chex
from math import ceil

from jsl.lds.kalman_filter import LDS


def _rk2(x0, M, nsteps, dt):
    """
    class-independent second-order Runge-Kutta method for linear systems

    Parameters
    ----------
    x0: array(state_size, )
        Initial state of the system
    M: array(state_size, K)
        Evolution matrix
    nsteps: int
        Total number of steps to integrate
    dt: float
        integration step size

    Returns
    -------
    array(nsteps, state_size)
        Integration history
    """
    def f(x): return M @ x
    input_dim, *_ = x0.shape

    def step(xt, t):
        k1 = f(xt)
        k2 = f(xt + dt * k1)
        xt = xt + dt * (k1 + k2) / 2
        return xt, xt

    steps = jnp.arange(nsteps)
    _, simulation = lax.scan(step, x0, steps)

    simulation = jnp.vstack([x0, simulation])
    return simulation

def sample(key: chex.PRNGKey,
           params: LDS,
           x0: chex.Array,
           T: float,
           nsamples: int,
           dt: float=0.01,
           noisy: bool=False):
    """
    Run the Kalman Filter algorithm. First, we integrate
    up to time T, then we obtain nsamples equally-spaced points. Finally,
    we transform the latent space to obtain the observations

    Parameters
    ----------
    params: LDS
        Linear Dynamical System object
    key: jax.random.PRNGKey
    x0: array(state_size)
        Initial state of simulation
    T: float
        Final time of integration
    nsamples: int
        Number of observations to take from the total integration
    dt: float
        integration step size
    noisy: bool
        Whether to (naively) add noise to the state space

    Returns
    -------
    * array(nsamples, state_size)
        State-space values
    * array(nsamples, obs_size)
        Observed-space values
    * int
        Number of observations skipped between one
        datapoint and the next
    """
    nsteps = ceil(T / dt)
    jump_size = ceil(nsteps / nsamples)
    correction = nsamples - ceil(nsteps / jump_size)
    nsteps += correction * jump_size

    key_state, key_obs = random.split(key)
    obs_size, state_size = params.C.shape
    state_noise = random.multivariate_normal(key_state, jnp.zeros(state_size), params.Q, (nsteps,))
    obs_noise = random.multivariate_normal(key_obs, jnp.zeros(obs_size), params.R, (nsteps,))
    simulation = _rk2(x0, params.A, nsteps, dt)

    if noisy:
        simulation = simulation + state_noise

    sample_state = simulation[::jump_size]
    sample_obs = jnp.einsum("ij,si->si", params.C, sample_state) + obs_noise[:len(sample_state)]

    return sample_state, sample_obs, jump_size

def filter(params: LDS,
           x_hist: chex.Array,
           jump_size: chex.Array,
           dt: chex.Array):
    """
    Compute the online version of the Kalman-Filter, i.e,
    the one-step-ahead prediction for the hidden state or the
    time update step

    Parameters
    ----------
    x_hist: array(timesteps, observation_size)

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
    obs_size, state_size = params.C.shape

    I = jnp.eye(state_size)
    timesteps, *_ = x_hist.shape
    mu_hist = jnp.zeros((timesteps, state_size))
    Sigma_hist = jnp.zeros((timesteps, state_size, state_size))
    Sigma_cond_hist = jnp.zeros((timesteps, state_size, state_size))
    mu_cond_hist = jnp.zeros((timesteps, state_size))

    # Initial configuration
    A, Q, C, R = params.A, params.Q, params.C, params.R
    mu, Sigma = params.mu, params.Sigma

    K1 = Sigma @ C.T @ inv(C @ Sigma @ C.T + R)
    mu1 = mu + K1 @ (x_hist[0] - C @ mu)
    Sigma1 = (I - K1 @ C) @ Sigma


    def rk_integration_step(state, carry):
        # Runge-kutta integration step
        mu, Sigma = state
        k1 = A @ mu
        k2 = A @ (mu + dt * k1)
        mu = mu + dt * (k1 + k2) / 2

        k1 = A @ Sigma @ A.T + Q
        k2 = A @ (Sigma + dt * k1) @ A.T + Q
        Sigma = Sigma + dt * (k1 + k2) / 2

        return (mu, Sigma), None

    def step(state, x):
        mun, Sigman = state
        initial_state = (mun, Sigman)
        (mun, Sigman), _ = lax.scan(rk_integration_step, initial_state, jnp.arange(jump_size))

        Sigman_cond = jnp.ones_like(Sigman) * Sigman
        St = C @ Sigman_cond @ C.T + R
        Kn = Sigman_cond @ C.T @ inv(St)

        mu_update = jnp.ones_like(mun) * mun
        x_update = C @ mun
        mun = mu_update + Kn @ (x - x_update)
        Sigman = (I - Kn @ C) @ Sigman_cond

        return (mun, Sigman), (mun, Sigman, mu_update, Sigman_cond)

    initial_state = (mu1, Sigma1)
    _, (mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist) = lax.scan(step, initial_state, x_hist)

    mu_hist = jnp.vstack([mu1[None, ...], mu_hist])
    Sigma_hist = jnp.vstack([Sigma1[None, ...], Sigma_hist])
    mu_cond_hist = jnp.vstack([params.mu[None, ...], mu_cond_hist])
    Sigma_cond_hist = jnp.vstack([params.Sigma[None, ...], Sigma_cond_hist])

    return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist
