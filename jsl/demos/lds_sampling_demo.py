from cProfile import label
import jax.numpy as jnp
from jax.random import  PRNGKey
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions
import matplotlib.pyplot  as plt

from jsl.lds.kalman_filter import LDS, kalman_filter
from jsl.lds.kalman_sampler import smooth_sampler


# Define the 1-d LDS model using the LDS model in the packate tensorflow_probability
ndims = 1
step_std = 1.0
noise_std = 5.0
model = tfd.LinearGaussianStateSpaceModel(
  num_timesteps=100,
  transition_matrix=tf.linalg.LinearOperatorDiag(jnp.array([1.01])),
  transition_noise=tfd.MultivariateNormalDiag(
   scale_diag=step_std * tf.ones([ndims])),
  observation_matrix=tf.linalg.LinearOperatorIdentity(ndims),
  observation_noise=tfd.MultivariateNormalDiag(
   scale_diag=noise_std * tf.ones([ndims])),
  initial_state_prior=tfd.MultivariateNormalDiag(loc=jnp.array([5.0]),
   scale_diag=tf.ones([ndims])))

# Sample from the prior of the LDS
y = model.sample() 
# Posterior sampling of the state variable using the built in method of the LDS model (for the sake of comparison)
smps = model.posterior_sample(y, sample_shape=50)
s_tf = jnp.array(smps[:,:,0])

# Define the same model as an LDS object defined in the kalman_filter file 
A = jnp.eye(1) * 1.01
C = jnp.eye(1)
Q = jnp.eye(1)
R = jnp.eye(1) * 25.0
mu0 = jnp.array([5.0])
Sigma0 = jnp.eye(1)
model_lds = LDS(A, C, Q, R, mu0, Sigma0)
# Run the Kalman filter algorithm first
mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = kalman_filter(model_lds, jnp.array(y))
# Sample backwards using the smoothing posterior
smooth_sample = smooth_sampler(model_lds, PRNGKey(0), mu_hist, Sigma_hist, n_samples=50)


# Plot the observation and posterior samples of state variables
plt.plot(y, color='red', label='Observation') # Observation
plt.plot(s_tf.T, alpha=0.12, color='blue') # Samples using TF built in function
plt.plot(smooth_sample[:,:,0].T, alpha=0.12, color='green') # Kalman smoother backwards sampler.
plt.legend()
plt.show()
