"""
Implementation of the Bootrstrap Filter for discrete time systems
**This implementation considers the case of multivariate normals**


"""
import jax.numpy as jnp
from jax import random, lax

import chex

from jax.scipy import stats
from jsl.nlds.base import NLDS


# TODO: Extend to general case
def filter(params: NLDS,
           key: chex.PRNGKey,
           init_state: chex.Array,
           sample_obs: chex.Array,
           nsamples: int = 2000,
           Vinit: chex.Array = None):
    """
    init_state: array(state_size,)
        Initial state estimate
    sample_obs: array(nsamples, obs_size)
        Samples of the observations
    """
    m, *_ = init_state.shape

    fx, fz = params.fx, params.fz
    Q, R = params.Qz, params.Rx

    key, key_init = random.split(key, 2)
    V = Q(init_state) if Vinit is None else Vinit
    zt_rvs = random.multivariate_normal(key_init, init_state, V, shape=(nsamples,))

    init_state = (zt_rvs, key)

    def __filter_step(state, obs_t):
        indices = jnp.arange(nsamples)
        zt_rvs, key_t = state

        key_t, key_reindex, key_next = random.split(key_t, 3)
        # 1. Draw new points from the dynamic model
        zt_rvs = random.multivariate_normal(key_t, fz(zt_rvs), Q(zt_rvs))

        # 2. Calculate unnormalised weights
        xt_rvs = fx(zt_rvs)
        weights_t = stats.multivariate_normal.pdf(obs_t, xt_rvs, R(zt_rvs, obs_t))

        # 3. Resampling
        pi = random.choice(key_reindex, indices,
                           p=weights_t, shape=(nsamples,))
        zt_rvs = zt_rvs[pi, ...]
        weights_t = jnp.ones(nsamples) / nsamples

        # 4. Compute latent-state estimate,
        #    Set next covariance state matrix
        mu_t = jnp.einsum("im,i->m", zt_rvs, weights_t)

        return (zt_rvs, key_next), mu_t

    _, mu_hist = lax.scan(__filter_step, init_state, sample_obs)

    return mu_hist
