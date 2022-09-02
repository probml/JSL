# Demo showcasing the training of an MLP with a single hidden layer using
# Extended Kalman Filtering (EKF).
# In this demo, we consider the latent state to be the weights of an MLP.
#   The observed state at time t is the output of the MLP as influenced by the weights
#   at time t-1 and the covariate x[t].
#   The transition function between latent states is the identity function.
# For more information, see
#   * Neural Network Training Using Unscented and Extended Kalman Filter
#       https://juniperpublishers.com/raej/RAEJ.MS.ID.555568.php

import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
import jsl.nlds.extended_kalman_filter as ekf_lib
from jax.flatten_util import ravel_pytree
from functools import partial
from typing import Sequence
from jsl.nlds.base import NLDS

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def apply(flat_params, x, model, unflatten_fn):
    """
    Multilayer Perceptron (MLP) with a single hidden unit and
    tanh activation function. The input-unit and the
    output-unit are both assumed to be unidimensional

    Parameters
    ----------
    W: array(2 * n_hidden + n_hidden + 1)
        Unravelled weights of the MLP
    x: array(1,)
        Singleton element to evaluate the MLP
    n_hidden: int
        Number of hidden units

    Returns
    -------
    * array(1,)
        Evaluation of MLP at the specified point
    """
    params = unflatten_fn(flat_params)
    return model.apply(params, x)


def sample_observations(key, f, n_obs, xmin, xmax, x_noise=0.1, y_noise=3.0):
    key_x, key_y, key_shuffle = jax.random.split(key, 3)
    x_noise = jax.random.normal(key_x, (n_obs,)) * x_noise
    y_noise = jax.random.normal(key_y, (n_obs,)) * y_noise
    x = jnp.linspace(xmin, xmax, n_obs) + x_noise
    y = f(x) + y_noise
    X = jnp.c_[x, y]

    shuffled_ixs = jax.random.permutation(key_shuffle, jnp.arange(n_obs))
    X, y = jnp.array(X[shuffled_ixs, :].T)
    return X, y


def plot_mlp_prediction(key, xobs, yobs, xtest, fw, w, Sw, ax, n_samples=100):
    W_samples = jax.random.multivariate_normal(key, w, Sw, (n_samples,))
    sample_yhat = fw(W_samples, xtest[:, None])
    for sample in sample_yhat:  # sample curves
        ax.plot(xtest, sample, c="tab:gray", alpha=0.07)
    ax.plot(xtest, sample_yhat.mean(axis=0))  # mean of posterior predictive
    ax.scatter(xobs, yobs, s=14, c="none", edgecolor="black", label="observations", alpha=0.5)
    ax.set_xlim(xobs.min(), xobs.max())


def plot_intermediate_steps(key, ax, fwd_func, intermediate_steps, xtest, mu_hist, Sigma_hist, x, y):
    """
    Plot the intermediate steps of the training process, all of them in the same plot
    but in different subplots.
    """
    for step, axi in zip(intermediate_steps, ax.flatten()):
        W_step, SW_step = mu_hist[step], Sigma_hist[step]
        x_step, y_step = x[:step], y[:step]
        plot_mlp_prediction(key, x_step, y_step, xtest, fwd_func, W_step, SW_step, axi)
        axi.set_title(f"step={step}")
    plt.tight_layout()


def plot_intermediate_steps_single(key, method, fwd_func, intermediate_steps, xtest, mu_hist, Sigma_hist, x, y):
    """
    Plot the intermediate steps of the training process, each one in a different plot.
    """
    figures = {}
    for step in intermediate_steps:
        W_step, SW_step = mu_hist[step], Sigma_hist[step]
        x_step, y_step = x[:step], y[:step]
        fig_step, axi = plt.subplots()
        plot_mlp_prediction(key, x_step, y_step, xtest, fwd_func, W_step, SW_step, axi)
        axi.set_title(f"step={step}")
        plt.tight_layout()
        figname = f"{method}-mlp-step-{step}"
        figures[figname] = fig_step
    return figures


def f(x):
    return x - 10 * jnp.cos(x) * jnp.sin(x) + x ** 3


def fz(W):
    return W


def main():
    key = jax.random.PRNGKey(314)
    key_sample_obs, key_weights, key_init = jax.random.split(key, 3)

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
    fwd_mlp_obs = jax.vmap(fwd_mlp, in_axes=[None, 0])
    # vectorised for multiple observations and weights
    fwd_mlp_obs_weights = jax.vmap(fwd_mlp_obs, in_axes=[0, None])

    # *** Generating training and test data ***
    n_obs = 200
    xmin, xmax = -3, 3
    sigma_y = 3.0
    x, y = sample_observations(key_sample_obs, f, n_obs, xmin, xmax, x_noise=0, y_noise=sigma_y)
    xtest = jnp.linspace(x.min(), x.max(), n_obs)

    # *** MLP Training with EKF ***
    n_params = W0.size
    W0 = jax.random.normal(key_weights, (n_params,)) * 1  # initial random guess
    Q = jnp.eye(n_params) * 1e-4  # parameters do not change
    R = jnp.eye(1) * sigma_y ** 2  # observation noise is fixed
    Vinit = jnp.eye(n_params) * 100  # vague prior

    ekf = NLDS(fz, fwd_mlp, Q, R)
    (W_ekf, SW_ekf), hist_ekf = ekf_lib.filter(ekf, W0, y[:, None], x[:, None], Vinit, return_params=["mean", "cov"])
    ekf_mu_hist, ekf_Sigma_hist = hist_ekf["mean"], hist_ekf["cov"]

    # Plot final performance
    fig, ax = plt.subplots()
    plot_mlp_prediction(key, x, y, xtest, fwd_mlp_obs_weights, W_ekf, SW_ekf, ax)
    ax.set_title("EKF + MLP")
    all_figures["ekf-mlp"] = fig

    # Plot intermediate performance
    intermediate_steps = [10, 20, 30, 40, 50, 60]
    fig, ax = plt.subplots(2, 2)
    plot_intermediate_steps(key, ax, fwd_mlp_obs_weights, intermediate_steps, xtest, ekf_mu_hist, ekf_Sigma_hist, x, y)
    plt.suptitle("EKF + MLP training")
    all_figures["ekf-mlp-intermediate"] = fig
    figures_intermediates = plot_intermediate_steps_single(key, "ekf", fwd_mlp_obs_weights,
                                                           intermediate_steps, xtest, ekf_mu_hist, ekf_Sigma_hist, x, y)
    all_figures = {**all_figures, **figures_intermediates}
    return all_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    figures = main()
    savefig(figures)
    plt.show()
