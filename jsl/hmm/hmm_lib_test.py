'''
This demo compares the Jax, Numpy and Distrax version of forwards-backwards algorithm in terms of the speed.
Also, checks whether or not they give the same result.
Author : Aleyna Kara (@karalleyna)
'''


import time

import jax.numpy as jnp
from jax.random import PRNGKey, split, uniform
import numpy as np
from jax import vmap, jit
from jax.random import split, randint, PRNGKey
import jax.numpy as jnp

from jsl.hmm.hmm_lib import HMMJax, HMMNumpy
from jsl.hmm.hmm_lib import hmm_sample_jax, hmm_forwards_backwards_jax, hmm_forwards_backwards_numpy
from jsl.hmm.hmm_lib import  hmm_loglikelihood_numpy, hmm_loglikelihood_jax
import jsl.hmm.hmm_utils as hmm_utils

import distrax
from distrax import HMM


#######
# Test log likelihood

def loglikelihood_numpy(params_numpy, batches, lens):
  return np.vstack([hmm_loglikelihood_numpy(params_numpy, batch, l) for batch, l in zip(batches, lens)])

def loglikelihood_jax(params_jax, batches, lens):
  return vmap(hmm_loglikelihood_jax, in_axes=(None, 0, 0))(params_jax, batches, lens)

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

assert np.allclose(ll_numpy, ll_jax)
print(f'Loglikelihood {ll_numpy}')

########
#Test Inference

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

assert np.allclose(alphas, alphas_jax, 8)
assert np.allclose(loglikelihood, loglikelihood_jax)
assert np.allclose(gammas, gammas_jax, 8)