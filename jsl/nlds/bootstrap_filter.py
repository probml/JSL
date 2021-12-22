import jax.numpy as jnp
from jax import random

class BootstrapFiltering(NLDS):
    def __init__(self, fz, fx, Q, R):
        """
        Implementation of the Bootrstrap Filter for discrete time systems
        **This implementation considers the case of multivariate normals**

        to-do: extend to general case
        """
        super().__init__(fz, fx, Q, R)
    
    def __filter_step(self, state, obs_t):
        nsamples = self.nsamples
        indices = jnp.arange(nsamples)
        zt_rvs, key_t = state

        key_t, key_reindex, key_next = random.split(key_t, 3)
        # 1. Draw new points from the dynamic model
        zt_rvs = random.multivariate_normal(key_t, self.fz(zt_rvs), self.Q(zt_rvs))

        # 2. Calculate unnormalised weights
        xt_rvs = self.fx(zt_rvs)
        weights_t = stats.multivariate_normal.pdf(obs_t, xt_rvs, self.R(zt_rvs, obs_t))

        # 3. Resampling
        pi = random.choice(key_reindex, indices,
                           p=weights_t, shape=(nsamples,))
        zt_rvs = zt_rvs[pi, ...]
        weights_t = jnp.ones(nsamples) / nsamples

        # 4. Compute latent-state estimate,
        #    Set next covariance state matrix
        mu_t = jnp.einsum("im,i->m", zt_rvs, weights_t)

        return (zt_rvs, key_next), mu_t


    def filter(self, key, init_state, sample_obs, nsamples=2000, Vinit=None):
            """
            init_state: array(state_size,)
                Initial state estimate
            sample_obs: array(nsamples, obs_size)
                Samples of the observations
            """
            m, *_ = init_state.shape
            nsteps = sample_obs.shape[0]
            mu_hist = jnp.zeros((nsteps, m))

            key, key_init = random.split(key, 2)
            V = self.Q(init_state) if Vinit is None else Vinit
            zt_rvs = random.multivariate_normal(key_init, init_state, V, shape=(nsamples,))
            
            init_state = (zt_rvs, key)
            self.nsamples = nsamples
            _, mu_hist = jax.lax.scan(self.__filter_step, init_state, sample_obs)

            return mu_hist
