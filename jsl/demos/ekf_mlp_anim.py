# Example showcasing the learning process of the EKF algorithm.
# This demo is based on the ekf_mlp_anim_demo.py demo.
# The animation script produces <a href="https://github.com/probml/probml-data/blob/main/data/ekf_mlp_demo.mp4">this video</a>.

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.random import PRNGKey, split, normal, multivariate_normal

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

from jsl.demos.ekf_mlp import MLP, apply, sample_observations
from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter


def main(fx, fz, filepath):
    key = PRNGKey(314)
    key_sample_obs, key_weights, key_init = split(key, 3)

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
    x, y = sample_observations(key_sample_obs, fx, n_obs, xmin, xmax, x_noise=0, y_noise=sigma_y)
    xtest = jnp.linspace(x.min(), x.max(), n_obs)

    # *** MLP Training with EKF ***
    n_params = W0.size
    W0 = normal(key_weights, (n_params,)) * 1  # initial random guess
    Q = jnp.eye(n_params) * 1e-4  # parameters do not change
    R = jnp.eye(1) * sigma_y ** 2  # observation noise is fixed
    Vinit = jnp.eye(n_params) * 100  # vague prior

    ekf = NLDS(fz, fwd_mlp, Q, R)
    _, ekf_hist = filter(ekf, W0, y[:, None], x[:, None], Vinit, return_params=["mean", "cov"])
    ekf_mu_hist, ekf_Sigma_hist = ekf_hist["mean"], ekf_hist["cov"]

    xtest = jnp.linspace(x.min(), x.max(), 200)
    fig, ax = plt.subplots()

    def func(i):
        plt.cla()
        W, SW = ekf_mu_hist[i], ekf_Sigma_hist[i]
        W_samples = multivariate_normal(key, W, SW, (100,))
        sample_yhat = fwd_mlp_obs_weights(W_samples, xtest[:, None])
        for sample in sample_yhat:
            ax.plot(xtest, sample, c="tab:gray", alpha=0.07)
        ax.plot(xtest, sample_yhat.mean(axis=0))
        ax.scatter(x[:i], y[:i], s=14, c="none", edgecolor="black", label="observations")
        ax.scatter(x[i], y[i], s=30, c="tab:red")
        ax.set_title(f"EKF+MLP ({i + 1:03}/{n_obs})")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        return ax

    ani = animation.FuncAnimation(fig, func, frames=n_obs)
    ani.save(filepath, dpi=200, bitrate=-1, fps=10)


if __name__ == "__main__":
    import os

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    path = os.environ.get("FIGDIR")
    path = "." if path is None else path
    filepath = os.path.join(path, "samples_hist_ekf.mp4")

    def f(x): return x - 10 * jnp.cos(x) * jnp.sin(x) + x ** 3
    def fz(W): return W
    main(f, fz, filepath)

    print(f"Saved animation to {filepath}")
