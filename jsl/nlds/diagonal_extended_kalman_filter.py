"""
Implementation of the Diagonal Extended Kalman Filter for a nonlinear
dynamical system with discrete observations. Also known as the
Node-decoupled Extended Kalman Filter (NDEKF)
"""

import jax.numpy as jnp
from jax import jacrev, lax

import chex
from typing import Tuple

from .base import NLDS


def filter(params: NLDS,
           init_state: chex.Array,
           sample_obs: chex.Array,
           observations: Tuple = None,
           Vinit: chex.Array = None):
    """
    Run the Extended Kalman Filter algorithm over a set of observed samples.
    Parameters
    ----------
    init_state: array(state_size)
    sample_obs: array(nsamples, obs_size)
    Returns
    -------
    * array(nsamples, state_size)
        History of filtered mean terms
    * array(nsamples, state_size, state_size)
        History of filtered covariance terms
    """
    state_size, *_ = init_state.shape

    fz, fx = params.fz, params.fx
    Q, R = params.Qz, params.Rx
    Dfx = jacrev(fx)

    Vt = Q(init_state) if Vinit is None else Vinit

    t = 0
    state = (init_state, Vt, t)
    observations = (observations,) if type(observations) is not tuple else observations
    xs = (sample_obs, observations)

    def filter_step(state, xs):
        """
        Run the Extended Kalman filter algorithm for a single step
        Paramters
        ---------
        state: tuple
            Mean, covariance at time t-1
        xs: tuple
            Target value and observations at time t
        """
        mu_t, Vt, t = state
        xt, obs = xs

        mu_t_cond = fz(mu_t)
        Ht = Dfx(mu_t_cond, *obs)

        Rt = R(mu_t_cond, *obs)
        xt_hat = fx(mu_t_cond, *obs)
        xi = xt - xt_hat
        A = jnp.linalg.inv(Rt + jnp.einsum("id,jd,d->ij", Ht, Ht, Vt))
        mu_t = mu_t_cond + jnp.einsum("s,is,ij,j->s", Vt, Ht, A, xi)
        Vt = Vt - jnp.einsum("s,is,ij,is,s->s", Vt, Ht, A, Ht, Vt) + Q(mu_t, t)

        return (mu_t, Vt, t + 1), (mu_t, None)

    (mu_t, Vt, _), mu_t_hist = lax.scan(filter_step, state, xs)

    return (mu_t, Vt), mu_t_hist
