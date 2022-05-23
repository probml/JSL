from jax import random
from jax import numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from jsl.lds.kalman_filter import LDS, kalman_filter, kalman_smoother


def lds_jsl_to_tfp(num_timesteps, lds):
    """Convert a JSL `LDS` object into a tfp `LinearGaussianStateSpaceModel`.

    Args:
        num_timesteps: int, number of timesteps.
        lds: LDS object.
    """
    dynamics_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=lds.Q)
    emission_noise_dist = tfd.MultivariateNormalFullCovariance(covariance_matrix=lds.R)
    initial_dist = tfd.MultivariateNormalFullCovariance(lds.mu, lds.Sigma)

    tfp_lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps,
        lds.A, dynamics_noise_dist,
        lds.C, emission_noise_dist,
        initial_dist,
    )

    return tfp_lgssm


def test_kalman_filter():
    key = random.PRNGKey(314)
    num_timesteps = 15
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
    z_hist, x_hist = lds_instance.sample(key, num_timesteps)

    filter_output = kalman_filter(lds_instance, x_hist)
    JSL_filtered_means, JSL_filtered_covs, *_ = filter_output
    JSL_smoothed_means, JSL_smoothed_covs = kalman_smoother(lds_instance, *filter_output)

    tfp_lgssm = lds_jsl_to_tfp(num_timesteps, lds_instance)
    _, tfp_filtered_means, tfp_filtered_covs, *_ = tfp_lgssm.forward_filter(x_hist)
    tfp_smoothed_means, tfp_smoothed_covs = tfp_lgssm.posterior_marginals(x_hist)

    assert np.allclose(JSL_filtered_means, tfp_filtered_means, rtol=1e-2)
    assert np.allclose(JSL_filtered_covs, tfp_filtered_covs, rtol=1e-2)
    assert np.allclose(JSL_smoothed_means, tfp_smoothed_means, rtol=1e-2)
    assert np.allclose(JSL_smoothed_covs, tfp_smoothed_covs, rtol=1e-2)
