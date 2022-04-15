# Demo showcasing the training of an MLP with a single hidden layer using 
# Unscented Kalman Filtering (UKF).
# In this demo, we consider the latent state to be the weights of an MLP.
#   The observed state at time t is the output of the MLP as influenced by the weights
#   at time t-1 and the covariate x[t].
#   The transition function between latent states is the identity function.
# For more information, see
#   * UKF-based training algorithm for feed-forward neural networks with
#     application to XOR classification problem
#       https://ieeexplore.ieee.org/document/6234549

import jax.numpy as jnp
from jax import vmap
from jax.random import PRNGKey, split, normal
from jax.flatten_util import ravel_pytree

import matplotlib.pyplot as plt
from functools import partial

import jsl.nlds.unscented_kalman_filter as ukf_lib
from jsl.nlds.base import NLDS
from jsl.demos.ekf_mlp import MLP, sample_observations, apply
from jsl.demos.ekf_mlp import plot_mlp_prediction, plot_intermediate_steps, plot_intermediate_steps_single


def f(x):
    return x - 10 * jnp.cos(x) * jnp.sin(x) + x ** 3


def fz(W):
    return W


def main():
    key = PRNGKey(314)
    key_sample_obs, key_weights, key_init = split(key, 3)

    all_figures = {}

    # *** MLP configuration ***
    n_hidden = 6
    n_out = 1
    n_in = 1
    model = MLP([n_hidden, n_out])

    batch_size = 20
    batch = jnp.ones((batch_size, n_in))

    variables = model.init(key_init, batch)
    W0, unflatten_fn = ravel_pytree(variables)

    fwd_mlp = partial(apply, model=model, unflatten_fn=unflatten_fn)
    # vectorised for multiple observations
    fwd_mlp_obs = vmap(fwd_mlp, in_axes=[None, 0])
    # vectorised for multiple weights
    fwd_mlp_weights = vmap(fwd_mlp, in_axes=[1, None])
    # vectorised for multiple observations and weights
    fwd_mlp_obs_weights = vmap(fwd_mlp_obs, in_axes=[0, None])

    # *** Generating training and test data ***
    n_obs = 200
    xmin, xmax = -3, 3
    sigma_y = 3.0
    x, y = sample_observations(key_sample_obs, f, n_obs, xmin, xmax, x_noise=0, y_noise=sigma_y)
    xtest = jnp.linspace(x.min(), x.max(), n_obs)

    # *** MLP Training with UKF ***
    n_params = W0.size
    W0 = normal(key_weights, (n_params,)) * 1  # initial random guess
    Q = jnp.eye(n_params) * 1e-4  # parameters do not change
    R = jnp.eye(1) * sigma_y ** 2  # observation noise is fixed

    Vinit = jnp.eye(n_params) * 5  # vague prior
    alpha, beta, kappa = 0.01, 2.0, 3.0 - n_params
    ukf = NLDS(fz, lambda w, x: fwd_mlp_weights(w, x).T, Q, R, alpha, beta, kappa, n_params)
    ukf_mu_hist, ukf_Sigma_hist = ukf_lib.filter(ukf, W0, y, x[:, None], Vinit)
    step = -1
    W_ukf, SW_ukf = ukf_mu_hist[step], ukf_Sigma_hist[step]

    fig, ax = plt.subplots()
    plot_mlp_prediction(key, x, y, xtest, fwd_mlp_obs_weights, W_ukf, SW_ukf, ax)
    ax.set_title("UKF + MLP")
    all_figures["ukf-mlp"] = fig

    fig, ax = plt.subplots(2, 2)
    intermediate_steps = [10, 20, 30, 40, 50, 60]
    plot_intermediate_steps(key, ax, fwd_mlp_obs_weights, intermediate_steps, xtest, ukf_mu_hist, ukf_Sigma_hist, x,
                            y)
    plt.suptitle("UKF + MLP training")
    all_figures["ukf-mlp-intermediate"] = fig
    figures_intermediate = plot_intermediate_steps_single(key, "ukf", fwd_mlp_obs_weights,
                                                          intermediate_steps, xtest, ukf_mu_hist, ukf_Sigma_hist, x,
                                                          y)
    all_figures = {**all_figures, **figures_intermediate}

    return all_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    figures = main()
    savefig(figures)
    plt.show()
