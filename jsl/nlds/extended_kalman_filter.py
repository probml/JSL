import chex
from jax import jacrev, lax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Callable
from functools import partial
from .base import NLDS

def filter_step(state: Tuple[chex.Array, chex.Array, int],
                xs: Tuple[chex.Array, chex.Array],
                params: NLDS,
                Dfx: Callable,
                Dfz: Callable,
                eps: float,
                return_params: Dict
                ) -> Tuple[Tuple[chex.Array, chex.Array, int], Dict]:
    """
    Run a single step of the extended Kalman filter (EKF) algorithm.

    Parameters
    ---------
    state: tuple
        Mean, covariance at time t-1
    xs: tuple
        Target value and covariates at time t
    params: NLDS
        Nonlinear dynamical system parameters
    Dfx: Callable
        Jacobian of the observation function
    Dfz: Callable
        Jacobian of the state transition function
    eps: float
        Small number to prevent singular matrix
    return_params: list
        Fix elements to carry

    Returns
    -------
    * tuple
        1. Mean, covariance, and time at time t
        2. History of filtered mean terms (if requested)
    """
    mu_t, Vt, t = state
    obs, inputs = xs

    state_size, *_ = mu_t.shape
    I = jnp.eye(state_size)
    Gt = Dfz(mu_t)
    mu_t_cond = params.fz(mu_t)
    Vt_cond = Gt @ Vt @ Gt.T + params.Qz(mu_t, t)
    Ht = Dfx(mu_t_cond, *inputs)

    Rt = params.Rx(mu_t_cond, *inputs)
    num_inputs, *_ = Rt.shape

    obs_hat = params.fx(mu_t_cond, *inputs)
    Mt = Ht @ Vt_cond @ Ht.T + Rt + eps * jnp.eye(num_inputs)
    Kt = Vt_cond @ Ht.T @ jnp.linalg.inv(Mt)
    mu_t = mu_t_cond + Kt @ (obs - obs_hat)
    Vt = (I - Kt @ Ht) @ Vt_cond @ (I - Kt @ Ht).T + Kt @ Rt @ Kt.T

    carry = {"mean": mu_t, "cov": Vt}
    carry = {key: val for key, val in carry.items() if key in return_params}
    return (mu_t, Vt, t + 1), carry


def filter(params: NLDS,
           init_state: chex.Array,
           observations: chex.Array,
           covariates: chex.Array = None,
           Vinit: chex.Array = None,
           return_params: List = None,
           eps: float = 0.001,
           return_history: bool = True):
    """
    Run the Extended Kalman Filter algorithm over a set of observed samples.

    Parameters
    ----------
    init_state: array(state_size)
    observations: array(nsamples, obs_size)
    covariates: array(nsamples, feature_size) or None
        optional covariates to pass to the observation function
    Vinit: array(state_size, state_size) or None
        Initial state covariance matrix
    return_params: list
        Parameters to carry from the filter step. Possible values are:
        "mean", "cov"
    return_history: bool
        Whether to return the history of mu and sigma obtained at each step

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

    Dfz = jacrev(fz)
    Dfx = jacrev(fx)

    Vt = Q(init_state) if Vinit is None else Vinit

    t = 0
    state = (init_state, Vt, t)
    covariates = (covariates,) if type(covariates) is not tuple else covariates
    xs = (observations, covariates)

    return_params = [] if return_params is None else return_params

    filter_step_pass = partial(filter_step, params=params, Dfx=Dfx, Dfz=Dfz,
                               eps=eps, return_params=return_params)
    (mu_t, Vt, _), hist_elements = lax.scan(filter_step_pass, state, xs)

    if return_history:
        return (mu_t, Vt), hist_elements

    return (mu_t, Vt), None
