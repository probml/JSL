import jax.numpy as jnp
from jax.random import multivariate_normal, PRNGKey
from jax.scipy.linalg import solve, cholesky
from jax import lax

from .kalman_filter import LDS



def smooth_sampler(params: LDS, 
                   key: PRNGKey,
                   mu_hist: jnp.array,
                   Sigma_hist: jnp.array,
                   n_samples: jnp.array = 1):
    """
    Backwards sample from the smoothing distribution
    Parameters
    ----------
    params: LDS
         Linear Dynamical System object
    key: jax.random.PRNGKey
            Seed of state noises
    mu_hist: array(timesteps, state_size):
        Filtered means mut
    Sigma_hist: array(timesteps, state_size, state_size)
        Filtered covariances Sigmat
    n_samples: int
        Number of posterior samples (optional)
    Returns
    -------
    * array(n_samples, timesteps, state_size):
        Posterior samples
    """
    state_size, _ = params.get_trans_mat_of(0).shape
    I = jnp.eye(state_size)
    timesteps = len(mu_hist)
    # Generate all state noise terms
    zeros_state = jnp.zeros(state_size)
    system_noise = multivariate_normal(key, zeros_state, I, (timesteps, n_samples))
    state_T = mu_hist[-1] + system_noise[-1] @ Sigma_hist[-1, ...].T

    def smooth_sample_step(state, inps):
        system_noise_t, mutt, Sigmatt , t =  inps
        A = params.get_trans_mat_of(t)
        et = state - mutt @ A.T 
        St = A @ Sigmatt @ A.T + params.get_system_noise_of(t)
        Kt = solve(St, A @ Sigmatt, sym_pos=True).T
        mu_t = mutt + et @ Kt.T
        Sigma_t = (I - Kt @ A) @ Sigmatt  
        Sigma_root = cholesky(Sigma_t)
        state_new = mu_t + system_noise_t @ Sigma_root.T
        return state_new, state_new
    
    inps = (system_noise[-2::-1, ...], mu_hist[-2::-1, ...], Sigma_hist[-2::-1, ...], jnp.arange(1, timesteps)[::-1])
    _, state_sample_smooth = lax.scan(smooth_sample_step, state_T, inps)

    state_sample_smooth = jnp.concatenate([state_sample_smooth[::-1, ...], state_T[None, ...]], axis=0)
    state_sample_smooth = jnp.swapaxes(state_sample_smooth, 0, 1)

    if n_samples == 1:
        state_sample_smooth = state_sample_smooth[:, 0, :]
    return state_sample_smooth