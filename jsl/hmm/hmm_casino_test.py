# Occasionally dishonest casino example [Durbin98, p54]. This script
# exemplifies a Hidden Markov Model (HMM) in which the throw of a die
# may result in the die being biased (towards 6) or unbiased. If the dice turns out to
# be biased, the probability of remaining biased is high, and similarly for the unbiased state.
# Assuming we observe the die being thrown n times the goal is to recover the periods in which
# the die was biased.
# Original matlab code: https://github.com/probml/pmtk3/blob/master/demos/casinoDemo.m



#from jsl.hmm.hmm_discrete_lib import (HMMNumpy, hmm_sample_numpy, hmm_plot_graphviz,
#                                  hmm_forwards_backwards_numpy, hmm_viterbi_numpy)

from jsl.hmm.hmm_utils import hmm_plot_graphviz

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import distrax
from distrax import HMM
from hmm_lib import HMMNumpy,  HMMJax, hmm_forwards_backwards_numpy, hmm_viterbi_numpy, hmm_viterbi_jax, hmm_forwards_backwards_jax
import hmm_logspace_lib

from jax.random import PRNGKey

def find_dishonest_intervals(z_hist):
    """
    Find the span of timesteps that the
    simulated systems turns to be in state 1
    Parameters
    ----------
    z_hist: array(n_samples)
        Result of running the system with two
        latent states
    Returns
    -------
    list of tuples with span of values
    """
    spans = []
    x_init = 0
    for t, _ in enumerate(z_hist[:-1]):
        if z_hist[t + 1] == 0 and z_hist[t] == 1:
            x_end = t
            spans.append((x_init, x_end))
        elif z_hist[t + 1] == 1 and z_hist[t] == 0:
            x_init = t + 1
    return spans

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

n_samples = 300
init_state_dist = jnp.array([1, 1]) / 2
hmm_numpy = HMMNumpy(np.array(A), np.array(B), np.array(init_state_dist))
hmm_jax = HMMJax(A,  B, init_state_dist)
hmm = HMM(trans_dist=distrax.Categorical(probs=A),
            init_dist=distrax.Categorical(probs=init_state_dist),
            obs_dist=distrax.Categorical(probs=B))
hmm_log = hmm_logspace_lib.HMM(trans_dist=distrax.Categorical(probs=A),
            init_dist=distrax.Categorical(probs=init_state_dist),
            obs_dist=distrax.Categorical(probs=B))

seed = 314
z_hist, x_hist = hmm.sample(seed=PRNGKey(seed), seq_len=n_samples)

z_hist_str = "".join((np.array(z_hist) + 1).astype(str))[:60]
x_hist_str = "".join((np.array(x_hist) + 1).astype(str))[:60]

print("Printing sample observed/latent...")
print(f"x: {x_hist_str}")
print(f"z: {z_hist_str}")



# Do inference
alpha_numpy, _, gamma_numpy, loglik_numpy = hmm_forwards_backwards_numpy(hmm_numpy,
                                                                         np.array(x_hist),
                                                                         len(x_hist))
alpha_jax, _, gamma_jax, loglik_jax = hmm_forwards_backwards_numpy(hmm_jax,
                                                                         x_hist,
                                                                         len(x_hist))

alpha_log, _, gamma_log, loglik_log = hmm_logspace_lib.hmm_forwards_backwards_log(hmm_log,
                                                                         x_hist,
                                                                         len(x_hist))
alpha, beta, gamma, loglik = hmm.forward_backward(x_hist)


assert np.allclose(alpha_numpy, alpha)
assert np.allclose(alpha_jax, alpha)
assert np.allclose(jnp.exp(alpha_log), alpha)

assert np.allclose(gamma_numpy, gamma)
assert np.allclose(gamma_jax, gamma)
assert np.allclose(jnp.exp(gamma_log), gamma)


print(f"Loglikelihood(Distrax): {loglik}")
print(f"Loglikelihood(Numpy): {loglik_numpy}")
print(f"Loglikelihood(Jax): {loglik_jax}")
print(f"Loglikelihood(Jax): {loglik_log}")

z_map_numpy = hmm_viterbi_numpy(hmm_numpy, np.array(x_hist))
z_map_jax = hmm_viterbi_jax(hmm_jax, x_hist)
z_map_log = hmm_logspace_lib.hmm_viterbi_log(hmm_log, x_hist)
z_map = hmm.viterbi(x_hist)

assert np.allclose(z_map_numpy, z_map)
assert np.allclose(z_map_jax, z_map)
assert np.allclose(z_map_log, z_map)
