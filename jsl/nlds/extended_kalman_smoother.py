# Extended Rauch-Tung-Striebel smoother or Extended Kalman Smoother (EKS)
import jax
import chex
import jax.numpy as jnp
from .base import NLDS
from functools import partial
from typing import Dict, List, Tuple, Callable
from jsl.nlds import extended_kalman_filter as ekf


def smooth_step(state: Tuple[chex.Array, chex.Array, int],
                xs: Tuple[chex.Array, chex.Array],
                params: NLDS,
                Dfz: Callable,
                eps: float,
                return_params: Dict
                ) -> Tuple[Tuple[chex.Array, chex.Array, int], Dict]:
    mean_next, cov_next, t = state
    mean_kf, cov_kf = xs

    mean_next_hat = params.fz(mean_kf)
    cov_next_hat = Dfz(mean_kf) @ cov_kf @ Dfz(mean_kf).T + params.Qz(mean_kf, t)
    kalman_gain = cov_kf @ Dfz(mean_kf).T @ jnp.linalg.inv(cov_next_hat + eps * jnp.eye(mean_next_hat.shape[0]))

    mean_prev = mean_kf + kalman_gain @ (mean_next - mean_next_hat)
    cov_prev = cov_kf + kalman_gain @ (cov_next - cov_next_hat) @ kalman_gain.T

    prev_state = (mean_prev, cov_prev, t-1)
    carry = {"mean": mean_prev, "cov": cov_prev}
    carry = {key: val for key, val in carry.items() if key in return_params}

    return prev_state, carry


def smooth(params: NLDS,
           init_state: chex.Array,
           observations: chex.Array,
           covariates: chex.Array = None,
           Vinit: chex.Array = None,
           return_params: List = None,
           eps: float = 0.001,
           ) -> Tuple[Tuple[chex.Array, chex.Array, int], Dict]:

    kf_params = ["mean", "cov"]
    Dfz = jax.jacrev(params.fz)
    _, ekf_hist = ekf.filter(params, init_state, observations, covariates, Vinit,
                            return_params=kf_params, eps=eps, return_history=True)
    kf_hist_mean, kf_hist_cov = ekf_hist["mean"], ekf_hist["cov"]
    kf_last_mean, kf_hist_mean = kf_hist_mean[-1], kf_hist_mean[:-1]
    kf_last_cov, kf_hist_cov = kf_hist_cov[-1], kf_hist_cov[:-1]

    smooth_step_partial =  partial(smooth_step, params=params, Dfz=Dfz,
                                   eps=eps, return_params=return_params)
    init_state = (kf_last_mean, kf_last_cov, len(kf_hist_mean) - 1)
    xs = (kf_hist_mean, kf_hist_cov)
    _, hist_smooth = jax.lax.scan(smooth_step_partial, init_state, xs, reverse=True)

    return hist_smooth
