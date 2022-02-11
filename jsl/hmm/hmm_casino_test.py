'''
Simple sanity check for all four Hidden Markov Models' implementations.
'''
import jax.numpy as jnp
from jax.random import PRNGKey

import matplotlib.pyplot as plt
import numpy as np

import distrax
from distrax import HMM

from jsl.hmm.hmm_numpy_lib import HMMNumpy, hmm_forwards_backwards_numpy, hmm_viterbi_numpy

from jsl.hmm.hmm_lib import HMMJax, hmm_viterbi_jax, hmm_forwards_backwards_jax

import jsl.hmm.hmm_logspace_lib as hmm_logspace_lib


def plot_inference(inference_values, z_hist, ax, state=1, map_estimate=False):
    """
    Plot the estimated smoothing/filtering/map of a sequence of hidden states.
    "Vertical gray bars denote times when the hidden
    state corresponded to state 1. Blue lines represent the
    posterior probability of being in that state given diﬀerent subsets
    of observed data." See Markov and Hidden Markov models section for more info
    Parameters
    ----------
    inference_values: array(n_samples, state_size)
        Result of runnig smoothing method
    z_hist: array(n_samples)
        Latent simulation
    ax: matplotlib.axes
    state: int
        Decide which state to highlight
    map_estimate: bool
        Whether to plot steps (simple plot if False)
    """
    n_samples = len(inference_values)
    xspan = np.arange(1, n_samples + 1)
    spans = find_dishonest_intervals(z_hist)
    if map_estimate:
        ax.step(xspan, inference_values, where="post")
    else:
        ax.plot(xspan, inference_values[:, state])

    for span in spans:
        ax.axvspan(*span, alpha=0.5, facecolor="tab:gray", edgecolor="none")
    ax.set_xlim(1, n_samples)
    # ax.set_ylim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Observation number")


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
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],  # fair die
    [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 5 / 10]  # loaded die
])

n_samples = 300
init_state_dist = jnp.array([1, 1]) / 2
hmm_numpy = HMMNumpy(np.array(A), np.array(B), np.array(init_state_dist))
hmm_jax = HMMJax(A, B, init_state_dist)
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
alpha_jax, _, gamma_jax, loglik_jax = hmm_forwards_backwards_jax(hmm_jax,
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

# Plot results
fig, ax = plt.subplots()
plot_inference(gamma_numpy, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Smoothed")
plt.savefig("hmm_casino_smooth_numpy.png")
plt.show()

fig, ax = plt.subplots()
plot_inference(z_map_numpy, z_hist, ax, map_estimate=True)
ax.set_ylabel("MAP state")
ax.set_title("Viterbi")
plt.savefig("hmm_casino_map_numpy.png")
plt.show()

# Plot results
fig, ax = plt.subplots()
plot_inference(gamma, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Smoothed")
# plt.savefig("hmm_casino_smooth_distrax.png")
plt.show()

fig, ax = plt.subplots()
plot_inference(z_map, z_hist, ax, map_estimate=True)
ax.set_ylabel("MAP state")
ax.set_title("Viterbi")
# plt.savefig("hmm_casino_map_distrax.png")
plt.show()

# Plot results
fig, ax = plt.subplots()
plot_inference(gamma_jax, z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Smoothed")
# plt.savefig("hmm_casino_smooth_jax.png")
plt.show()

fig, ax = plt.subplots()
plot_inference(z_map_jax, z_hist, ax, map_estimate=True)
ax.set_ylabel("MAP state")
ax.set_title("Viterbi")
# plt.savefig("hmm_casino_map_jax.png")
plt.show()

# Plot results
fig, ax = plt.subplots()
plot_inference(jnp.exp(gamma_log), z_hist, ax)
ax.set_ylabel("p(loaded)")
ax.set_title("Smoothed")
# plt.savefig("hmm_casino_smooth_log.png")
plt.show()

fig, ax = plt.subplots()
plot_inference(z_map_log, z_hist, ax, map_estimate=True)
ax.set_ylabel("MAP state")
ax.set_title("Viterbi")
# plt.savefig("hmm_casino_map_log.png")
plt.show()
