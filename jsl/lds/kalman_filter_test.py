from jax import random
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from jsl.lds.kalman_filter import LDS, kalman_filter

def tfp_filter(timesteps, A, transition_noise_scale, C, observation_noise_scale, mu0, x_hist):
    """ Perform filtering using tensorflow probability """
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
    return filtered_means.numpy(), filtered_covs.numpy()


def test_kalman_filter():
    key = random.PRNGKey(314)
    timesteps = 15 
    delta = 1.0

    ### LDS Parameters ###
    state_size = 2
    observation_size = 2
    A = jnp.eye(state_size)
    C = jnp.eye(state_size)

    transition_noise_scale = 1.0
    observation_noise_scale = 1.0
    Q = jnp.eye(state_size) * transition_noise_scale
    R = jnp.eye(observation_size) * observation_noise_scale


    ### Prior distribution params ###
    mu0 = jnp.array([8, 10]).astype(float)
    Sigma0 = jnp.eye(state_size) * 1.0

    ### Sample data ###
    lds_instance = LDS(A, C, Q, R, mu0, Sigma0)
    z_hist, x_hist = lds_instance.sample(key, timesteps)

    JSL_z_filt, JSL_Sigma_filt, _, _ = kalman_filter(lds_instance, x_hist)
    tfp_z_filt, tfp_Sigma_filt = tfp_filter(
        timesteps, A, transition_noise_scale, C, observation_noise_scale, mu0, x_hist
    )

    assert np.allclose(JSL_z_filt, tfp_z_filt, rtol=1e-2)
    assert np.allclose(JSL_Sigma_filt, tfp_Sigma_filt, rtol=1e-2)

