"""
Implementation of the Unscented Kalman Filter for discrete time systems
"""

import jax.numpy as jnp
from jax.lax import scan

import chex
from typing import List

from .base import NLDS


def sqrtm(M):
    """
    Compute the matrix square-root of a hermitian
    matrix M. i,e, R such that RR = M

    Parameters
    ----------
    M: array(m, m)
        Hermitian matrix

    Returns
    -------
    array(m, m): square-root matrix
    """
    evals, evecs = jnp.linalg.eigh(M)
    R = evecs @ jnp.sqrt(jnp.diag(evals)) @ jnp.linalg.inv(evecs)
    return R


def filter(params: NLDS,
           init_state: chex.Array,
           sample_obs: chex.Array,
           observations: List = None,
           Vinit: chex.Array = None,
           return_history: bool = True):
    """
    Run the Unscented Kalman Filter algorithm over a set of observed samples.
    Parameters
    ----------
    sample_obs: array(nsamples, obs_size)
    return_history: bool
        Whether to return the history of mu and Sigma values.
    Returns
    -------
    * array(nsamples, state_size)
        History of filtered mean terms
    * array(nsamples, state_size, state_size)
        History of filtered covariance terms
    """
    alpha = params.alpha
    beta = params.beta
    kappa = params.kappa
    d = params.d

    fx, fz = params.fx, params.fz
    Q, R = params.Qz, params.Rx

    lmbda = alpha ** 2 * (d + kappa) - d
    gamma = jnp.sqrt(d + lmbda)

    wm_vec = jnp.array([1 / (2 * (d + lmbda)) if i > 0
                        else lmbda / (d + lmbda)
                        for i in range(2 * d + 1)])
    wc_vec = jnp.array([1 / (2 * (d + lmbda)) if i > 0
                        else lmbda / (d + lmbda) + (1 - alpha ** 2 + beta)
                        for i in range(2 * d + 1)])
    nsteps, *_ = sample_obs.shape
    initial_mu_t = init_state
    initial_Sigma_t = Q(init_state) if Vinit is None else Vinit

    if observations is None:
        observations = iter([()] * nsteps)
    else:
        observations = iter([(obs,) for obs in observations])

    def filter_step(params, sample_observation):
        mu_t, Sigma_t = params
        observation = next(observations)

        # TO-DO: use jax.scipy.linalg.sqrtm when it gets added to lib
        comp1 = mu_t[:, None] + gamma * sqrtm(Sigma_t)
        comp2 = mu_t[:, None] - gamma * sqrtm(Sigma_t)
        # sigma_points = jnp.c_[mu_t, comp1, comp2]
        sigma_points = jnp.concatenate((mu_t[:, None], comp1, comp2), axis=1)

        z_bar = fz(sigma_points)
        mu_bar = z_bar @ wm_vec
        Sigma_bar = (z_bar - mu_bar[:, None])
        Sigma_bar = jnp.einsum("i,ji,ki->jk", wc_vec, Sigma_bar, Sigma_bar) + Q(mu_t)

        Sigma_bar_half = sqrtm(Sigma_bar)
        comp1 = mu_bar[:, None] + gamma * Sigma_bar_half
        comp2 = mu_bar[:, None] - gamma * Sigma_bar_half
        # sigma_points = jnp.c_[mu_bar, comp1, comp2]
        sigma_points = jnp.concatenate((mu_bar[:, None], comp1, comp2), axis=1)

        x_bar = fx(sigma_points, *observation)
        x_hat = x_bar @ wm_vec
        St = x_bar - x_hat[:, None]
        St = jnp.einsum("i,ji,ki->jk", wc_vec, St, St) + R(mu_t, *observation)

        mu_hat_component = z_bar - mu_bar[:, None]
        x_hat_component = x_bar - x_hat[:, None]
        Sigma_bar_y = jnp.einsum("i,ji,ki->jk", wc_vec, mu_hat_component, x_hat_component)
        Kt = Sigma_bar_y @ jnp.linalg.inv(St)

        mu_t = mu_bar + Kt @ (sample_observation - x_hat)
        Sigma_t = Sigma_bar - Kt @ St @ Kt.T

        return (mu_t, Sigma_t), (mu_t, Sigma_t)

    (mu, Sigma), (mu_hist, Sigma_hist) = scan(filter_step, (initial_mu_t, initial_Sigma_t), sample_obs[1:])

    mu_hist = jnp.vstack([initial_mu_t[None, ...], mu_hist])
    Sigma_hist = jnp.vstack([initial_Sigma_t[None, ...], Sigma_hist])

    if return_history:
        return mu_hist, Sigma_hist
    return mu, Sigma
