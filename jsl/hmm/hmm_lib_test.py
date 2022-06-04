'''
This demo compares the Jax, Numpy and Distrax version of forwards-backwards algorithm in terms of the speed.
Also, checks whether or not they give the same result.
Author : Aleyna Kara (@karalleyna)
'''

import jax.numpy as jnp
from jax import vmap, nn
from jax.random import split, PRNGKey, uniform, normal

import distrax
from distrax import HMM

import chex

import numpy as np
import time

from jsl.hmm.hmm_numpy_lib import HMMNumpy, hmm_forwards_backwards_numpy, hmm_loglikelihood_numpy
from jsl.hmm.hmm_lib import HMMJax, hmm_viterbi_jax
from jsl.hmm.hmm_lib import hmm_sample_jax, hmm_forwards_backwards_jax, hmm_loglikelihood_jax
from jsl.hmm.hmm_lib import normalize, fixed_lag_smoother
import jsl.hmm.hmm_utils as hmm_utils


from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

#######
# Test log likelihood

def loglikelihood_numpy(params_numpy, batches, lens):
  return np.array([hmm_loglikelihood_numpy(params_numpy, batch, l) for batch, l in zip(batches, lens)])

def loglikelihood_jax(params_jax, batches, lens):
  return vmap(hmm_loglikelihood_jax, in_axes=(None, 0, 0))(params_jax, batches, lens)[:,:, 0]


def test_all_hmm_models():
  # state transition matrix
  A = jnp.array([
      [0.95, 0.05],
      [0.10, 0.90]
  ])

  # observation matrix
  B = jnp.array([
      [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
      [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
  ])

  pi = jnp.array([1, 1]) / 2

  params_numpy= HMMNumpy(np.array(A), np.array(B), np.array(pi))
  params_jax = HMMJax(A, B, pi)

  seed = 0
  rng_key = PRNGKey(seed)
  rng_key, rng_sample = split(rng_key)

  n_obs_seq, batch_size, max_len = 15, 5, 10

  observations, lens = hmm_utils.hmm_sample_n(params_jax,
                                              hmm_sample_jax,
                                              n_obs_seq, max_len,
                                              rng_sample)

  observations, lens = hmm_utils.pad_sequences(observations, lens)

  rng_key, rng_batch = split(rng_key)
  batches, lens = hmm_utils.hmm_sample_minibatches(observations,
                                                  lens,
                                                  batch_size,
                                                  rng_batch)

  ll_numpy = loglikelihood_numpy(params_numpy, np.array(batches), np.array(lens))
  ll_jax = loglikelihood_jax(params_jax, batches, lens)
  assert np.allclose(ll_numpy, ll_jax, atol=4)


def test_inference():
  seed = 0
  rng_key = PRNGKey(seed)
  rng_key, key_A, key_B = split(rng_key, 3)

  # state transition matrix
  n_hidden, n_obs = 100, 10
  A = uniform(key_A, (n_hidden, n_hidden))
  A = A / jnp.sum(A, axis=1)

  # observation matrix
  B = uniform(key_B, (n_hidden, n_obs))
  B = B / jnp.sum(B, axis=1).reshape((-1, 1))

  n_samples = 1000
  init_state_dist = jnp.ones(n_hidden) / n_hidden

  seed = 0
  rng_key = PRNGKey(seed)

  params_numpy = HMMNumpy(A, B, init_state_dist)
  params_jax = HMMJax(A, B, init_state_dist)
  hmm_distrax = HMM(trans_dist=distrax.Categorical(probs=A),
                    obs_dist=distrax.Categorical(probs=B),
                    init_dist=distrax.Categorical(probs=init_state_dist))

  z_hist, x_hist = hmm_sample_jax(params_jax, n_samples, rng_key)

  start = time.time()
  alphas_np, _, gammas_np, loglikelihood_np = hmm_forwards_backwards_numpy(params_numpy, x_hist, len(x_hist))
  print(f'Time taken by numpy version of forwards backwards : {time.time()-start}s')

  start = time.time()
  alphas_jax, _, gammas_jax, loglikelihood_jax = hmm_forwards_backwards_jax(params_jax, jnp.array(x_hist), len(x_hist))
  print(f'Time taken by JAX version of forwards backwards: {time.time()-start}s')

  start = time.time()
  alphas, _, gammas, loglikelihood = hmm_distrax.forward_backward(obs_seq=jnp.array(x_hist),
                                                                  length=len(x_hist))

  print(f'Time taken by HMM distrax : {time.time()-start}s')

  assert np.allclose(alphas_np, alphas_jax)
  assert np.allclose(loglikelihood_np, loglikelihood_jax)
  assert np.allclose(gammas_np, gammas_jax)

  assert np.allclose(alphas, alphas_jax,  atol=8)
  assert np.allclose(loglikelihood, loglikelihood_jax, atol=8)
  assert np.allclose(gammas, gammas_jax,   atol=8)


def _make_models(init_probs, trans_probs, obs_probs, length):
  """Build distrax HMM and equivalent TFP HMM."""
  
  dx_model = HMMJax(
      trans_probs,
      obs_probs,
      init_probs
  )

  tfp_model = tfd.HiddenMarkovModel(
      initial_distribution=tfd.Categorical(probs=init_probs),
      transition_distribution=tfd.Categorical(probs=trans_probs),
      observation_distribution=tfd.Categorical(probs=obs_probs),
      num_steps=length,
  )

  return dx_model, tfp_model


def test_sample(length, num_states):
  params_fn = obs_dist_name_and_params_fn
  
  init_probs = nn.softmax(normal(PRNGKey(0), (num_states,)), axis=-1)
  trans_mat = nn.softmax(normal(PRNGKey(1), (num_states, num_states)), axis=-1)

  model, tfp_model = _make_models(init_probs,
                                  trans_mat,
                                  params_fn(num_states),
                                  length)

  states, obs = hmm_sample_jax(model, length, PRNGKey(0))
  tfp_obs = tfp_model.sample(seed=PRNGKey(0))

  chex.assert_shape(states, (length,))
  chex.assert_equal_shape([obs, tfp_obs])


def test_forward_backward(length, num_states):
  params_fn = obs_dist_name_and_params_fn
  
  init_probs = nn.softmax(normal(PRNGKey(0), (num_states,)), axis=-1)
  trans_mat = nn.softmax(normal(PRNGKey(1), (num_states, num_states)), axis=-1)

  model, tfp_model = _make_models(init_probs,
                                  trans_mat,
                                  params_fn(num_states),
                                  length)

  _, observations = hmm_sample_jax(model, length, PRNGKey(42))

  alphas, betas, marginals, log_prob = hmm_forwards_backwards_jax(model,
      observations)
  
  tfp_marginal_logits = tfp_model.posterior_marginals(observations).logits
  tfp_marginals = nn.softmax(tfp_marginal_logits)

  chex.assert_shape(alphas, (length, num_states))
  chex.assert_shape(betas, (length, num_states))
  chex.assert_shape(marginals, (length, num_states))
  chex.assert_shape(log_prob, (1,))
  np.testing.assert_array_almost_equal(marginals, tfp_marginals, decimal=4)


def test_viterbi(length, num_states):
  params_fn = obs_dist_name_and_params_fn
  
  init_probs = nn.softmax(normal(PRNGKey(0), (num_states,)), axis=-1)
  trans_mat = nn.softmax(normal(PRNGKey(1), (num_states, num_states)), axis=-1)

  model, tfp_model = _make_models(init_probs,
                                  trans_mat,
                                  params_fn(num_states),
                                  length)

  _, observations = hmm_sample_jax(model, length, PRNGKey(42))
  most_likely_states = hmm_viterbi_jax(model, observations)
  tfp_mode = tfp_model.posterior_mode(observations)
  chex.assert_shape(most_likely_states, (length,))
  assert np.allclose(most_likely_states, tfp_mode)

'''
########
#Test Fixed Lag Smoother

# helper function
def get_fls_result(params, data, win_len, act=None):
  assert data.size > 2, "Complete observation set must be of size at least 2"
  prior, obs_mat = params.init_dist, params.obs_mat
  n_states = obs_mat.shape[0]
  alpha, _ = normalize(prior * obs_mat[:, data[0]])
  bmatrix = jnp.eye(n_states)[None, :]
  for obs in data[1:]:
    alpha, bmatrix, gamma = fixed_lag_smoother(params, win_len, alpha, bmatrix, obs)
  return alpha, gamma

*_, gammas_fls = get_fls_result(params_jax, jnp.array(x_hist), jnp.array(x_hist).size)

assert np.allclose(gammas_fls, gammas_jax)
'''
obs_dist_name_and_params_fn = lambda n: nn.softmax(normal(PRNGKey(0), (n, 7)), axis=-1)

### Tests
test_all_hmm_models()
test_inference()

for length, num_states in zip([1, 3],  (2, 23)):
  test_viterbi(length, num_states)
  test_forward_backward(length, num_states)
  test_sample(length, num_states)


