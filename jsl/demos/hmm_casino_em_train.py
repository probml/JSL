"""
This demo shows the parameter estimations of HMMs via Baulm-Welch algorithm on the occasionally dishonest casino example.
Author : Aleyna Kara(@karalleyna)
"""


import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.random import split, PRNGKey
from jsl.hmm.hmm_numpy_lib import HMMNumpy, hmm_em_numpy
from jsl.hmm.hmm_lib import HMMJax, hmm_sample_jax
from jsl.hmm.hmm_lib import init_random_params_jax, hmm_em_jax
from jsl.hmm import hmm_utils


def main():
    A = jnp.array([
        [0.95, 0.05],
        [0.10, 0.90]
    ])

    # observation matrix
    B = jnp.array([
        [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],  # fair die
        [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 5 / 10]  # loaded die
    ])

    pi = jnp.array([1, 1]) / 2

    seed = 100
    rng_key = PRNGKey(seed)
    rng_key, rng_sample, rng_batch, rng_init = split(rng_key, 4)

    casino = HMMJax(A, B, pi)

    n_obs_seq, batch_size, max_len = 5, 5, 3000

    observations, lens = hmm_utils.hmm_sample_n(casino,
                                                hmm_sample_jax,
                                                n_obs_seq, max_len,
                                                rng_sample)
    observations, lens = hmm_utils.pad_sequences(observations, lens)

    n_hidden, n_obs = B.shape
    params_jax = init_random_params_jax([n_hidden, n_obs], rng_key=rng_init)
    params_numpy = HMMNumpy(np.array(params_jax.trans_mat),
                            np.array(params_jax.obs_mat),
                            np.array(params_jax.init_dist))

    num_epochs = 20

    start = time.time()
    params_numpy, neg_ll_numpy = hmm_em_numpy(np.array(observations),
                                              np.array(lens),
                                              num_epochs=num_epochs,
                                              init_params=params_numpy)
    print(f"Time taken by numpy version of EM : {time.time() - start}s")

    start = time.time()
    params_jax, neg_ll_jax = hmm_em_jax(observations,
                                        lens,
                                        num_epochs=num_epochs,
                                        init_params=params_jax)
    print(f"Time taken by JAX version of EM : {time.time() - start}s")

    assert jnp.allclose(np.array(neg_ll_jax), np.array(neg_ll_numpy), 4)

    print(f" Negative loglikelihoods : {neg_ll_jax}")

    dict_figures = {}
    fig, ax = plt.subplots()
    ax.plot(neg_ll_numpy, label="EM numpy")
    ax.set_label("Number of Iterations")
    dict_figures["em-numpy"] = fig

    fig, ax = plt.subplots()
    ax.plot(neg_ll_jax, label="EM JAX")
    ax.set_xlabel("Number of Iterations")
    dict_figures["em-jax"] = fig

    states, observations = ["Fair Dice", "Loaded Dice"], [str(i + 1) for i in range(B.shape[1])]
    # dotfile_np = hmm_plot_graphviz(params_numpy, "hmm_casino_train_np", states, observations)
    dotfile_np = hmm_utils.hmm_plot_graphviz(params_numpy.trans_mat, params_numpy.obs_mat,
                                             states, observations)

    # dotfile_jax = hmm_plot_graphviz(params_jax, "hmm_casino_train_jax", states, observations)
    dotfile_jax = hmm_utils.hmm_plot_graphviz(params_jax.trans_mat, params_jax.obs_mat,
                                              states, observations)

    dotfile_dict = {"graph-numpy": dotfile_np, "graph-jax": dotfile_jax}

    return dict_figures, dotfile_dict


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig, savedotfile

    figs, dotfiles = main()
    savefig(figs)
    savedotfile(dotfiles)
    plt.show()
