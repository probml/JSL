# Jax implementation of a Linear Dynamical System where the observation noise is not known.
# Author:  Gerardo Durán-Martín (@gerdm), Aleyna Kara(@karalleyna)

import chex

import jax.numpy as jnp
from jax import lax, vmap, tree_map

from dataclasses import dataclass
from functools import partial
from typing import Union, Callable

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


@dataclass
class LDS:
    """
    Kalman filtering for a linear Gaussian state space model with scalar observations,
    where all the parameters are known except for the observation noise variance.
    The model has the following form:
       p(state(0)) = Gauss(mu0, Sigma0)
       p(state(t) | state(t-1)) = Gauss(A * state(t-1), Q)
       p(obs(t) | state(t)) = Gauss(C * state(t), r)
    The value of r is jointly inferred together with the latent states to produce the posterior
    p(state(t) , r | obs(1:t)) = Gauss(mu(t), Sigma(t) * r) * Ga(1/r | nu(t)/2, nu(t)*tau(t)/2)
    where 1/r is the observation precision. For details on this algorithm, see sec 4.5 of
    "Bayesian forecasting and dynamic models", West and Harrison, 1997.
    https://www2.stat.duke.edu/~mw/West&HarrisonBook/
    https://bayanbox.ir/view/5561099385628144678/Bayesian-forecasting-and-dynamic-models-West-Harison.pdf

    Parameters
    ----------
    A: array(state_size, state_size)
        Transition matrix
    C: array(observation_size, state_size)
        Observation matrix
    Q: array(state_size, state_size)
        Transition covariance matrix
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
    v: chex.Array
    tau: chex.Array


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

    def kalman_step(state, obs):
        mu, Sigma, v, tau = state
        covariates, response = obs

        mu_cond = jnp.matmul(A, mu, precision=lax.Precision.HIGHEST)
        Sigmat_cond = jnp.matmul(jnp.matmul(A, Sigma, precision=lax.Precision.HIGHEST), A,
                                 precision=lax.Precision.HIGHEST) + Q

        e_k = response - covariates.T @ mu_cond
        s_k = covariates.T @ Sigmat_cond @ covariates + 1
        Kt = (Sigmat_cond @ covariates) / s_k

        mu = mu + e_k * Kt
        Sigma = Sigmat_cond - jnp.outer(Kt, Kt) * s_k

        v_update = v + 1
        tau = (v * tau + (e_k * e_k) / s_k) / v_update

        return (mu, Sigma, v_update, tau), (mu, Sigma)

    mu0, Sigma0 = params.mu, params.Sigma
    initial_state = (mu0, Sigma0, 0)
    (mu, Sigma, _, _), history = lax.scan(kalman_step, initial_state, x_hist)
    if return_history:
        return history
    return mu, Sigma


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
        return tree_map(lambda x: x[0, ...], outputs)
    return outputs
