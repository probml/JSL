# Extended Rauch-Tung-Striebel smoother or Extended Kalman Smoother (EKS)
import jax
import chex
import jax.numpy as jnp
from .base import NLDS
from typing import Dict, List, Tuple, Callable
from jax.nlds import extended_kalman_filter as ekf


def smooth_step(state: Tuple[chex.Array, chex.Array, int],
                xs: Tuple[chex.Array, chex.Array],
                params: NLDS,
                Dfz: Callable,
                eps: float,
                return_params: Dict
                ) -> Tuple[Tuple[chex.Array, chex.Array, int], Dict]:
    mean_next, cov_next, t = state
    mean_kf, cov_kf = xs

    mean_next_hat = NLDS.fz(mean_kf)
    cov_next_hat = Dfz(mean_kf) @ cov_kf @ Dfz(mean_kf).T + params.Qz(mean_kf, t)
    kalman_gain = cov_kf @ Dfz(mean_kf).T @ jnp.linalg.inv(cov_next_hat + eps * jnp.eye(mean_next_hat.shape[0]))

    mean_prev = mean_kf + kalman_gain @ (mean_next - mean_next_hat)
    cov_prev = cov_kf + kalman_gain @ cov_next @ kalman_gain.T

    prev_state = (mean_prev, cov_prev, t-1)
    carry = {"mean": mu_t, "cov": Vt}
    carry = {key: val for key, val in carry.items() if key in return_params}

    return prev_state, carry


def smooth(params: NLDS,
           init_state: chex.Array,
           sample_obs: chex.Array,
           covariates: chex.Array = None,
           Vinit: chex.Array = None,
           return_params: List = None,
           eps: float = 0.001,
           return_history: bool = True
           ) -> Tuple[Tuple[chex.Array, chex.Array, int], Dict]:
   ... 