from jsl.lds.kalman_sampler import smooth_sampler

from jax import random
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from jsl.lds.kalman_filter import LDS, kalman_filter

import pytest

class TestKalmanSmoother():
    # Utility functions
    def tfp_smoother(self, timesteps, A, transition_noise_scale, C, observation_noise_scale, mu0, x_hist):
        state_size, _ = A.shape
        observation_size, _ = C.shape
        transition_noise = tfd.MultivariateNormalDiag(
            scale_diag=jnp.ones(state_size) * transition_noise_scale
        )
        obs_noise = tfd.MultivariateNormalDiag(
            scale_diag=jnp.ones(observation_size) * observation_noise_scale
        )
        prior = tfd.MultivariateNormalDiag(mu0, tf.ones([state_size]))

        LGSSM = tfd.LinearGaussianStateSpaceModel(
            timesteps, A, transition_noise, C, obs_noise, prior
        )

        _, filtered_means, filtered_covs, _, _, _, _ = LGSSM.forward_filter(x_hist)
        smps = LGSSM.posterior_sample(x_hist, sample_shape=x_hist.shape[0])
        return jnp.array(smps[:,:,0])


    def LDS_instance(self, timesteps, A, C, Q, R, mu0, Sigma0):
        return LDS(A, C, Q, R, mu0, Sigma0)

    def jsl_smoother(self, lds_instance, key, x_hist):
        JSL_z_filt, JSL_Sigma_filt, _, _ = kalman_filter(lds_instance, x_hist)
        s_jax = smooth_sampler(lds_instance, key, JSL_z_filt, JSL_Sigma_filt, n_samples=x_hist.shape[0])[:,:,0]
        return s_jax
    
    # Test Kalman Smoother
    def test_kalman_smoother(self):
        timesteps = 15
        key = random.PRNGKey(0)
        observation_noise_scale = 1.0
        transition_noise_scale = 1.0
        
        A = jnp.eye(2)
        C = jnp.eye(2)

        Q = jnp.eye(2) * transition_noise_scale
        R = jnp.eye(2) * observation_noise_scale

        mu0 = jnp.array([5.0, 5.0])
        Sigma0 = jnp.eye(2) * 1.0

        lds = self.LDS_instance(timesteps, A, C, Q, R, mu0, Sigma0)

        z_hist, x_hist = lds.sample(key, timesteps)

        s_tf = self.tfp_smoother(timesteps, A, transition_noise_scale, C, observation_noise_scale, mu0, x_hist)
        s_jax = self.jsl_smoother(lds, key, x_hist)

        mean_tf = jnp.mean(s_tf, axis=0)
        mean_jax = jnp.mean(s_jax, axis=0)
        
        print(jnp.max(mean_tf - mean_jax))
        assert jnp.allclose(mean_tf, mean_jax, atol=1)
