# Example showcasing the learning process of the EKF algorithm.
# This demo is based on the ekf_mlp_anim_demo.py demo.
# The animation script produces <a href="https://github.com/probml/probml-data/blob/main/data/ekf_mlp_demo.mp4">this video</a>.

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jsl.demos import ekf_vs_ukf_mlp as demo
import matplotlib.animation as animation
from functools import partial
from jax.random import PRNGKey, split, normal, multivariate_normal

from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter

from celluloid import Camera

def main(fx, fz, filepath):
    # *** MLP configuration ***
    n_hidden = 6
    n_in, n_out = 1, 1
    n_params = (n_in + 1) * n_hidden + (n_hidden + 1) * n_out
    fwd_mlp = partial(demo.mlp, n_hidden=n_hidden)
    # vectorised for multiple observations
    fwd_mlp_obs = jax.vmap(fwd_mlp, in_axes=[None, 0])
    # vectorised for multiple weights
    fwd_mlp_weights = jax.vmap(fwd_mlp, in_axes=[1, None])
    # vectorised for multiple observations and weights
    fwd_mlp_obs_weights = jax.vmap(fwd_mlp_obs, in_axes=[0, None])

    # *** Generating training and test data ***
    n_obs = 200
    key = PRNGKey(314)
    key_sample_obs, key_weights = split(key, 2)
    xmin, xmax = -3, 3
    sigma_y = 3.0
    x, y = demo.sample_observations(key_sample_obs, fx, n_obs, xmin, xmax, x_noise=0, y_noise=sigma_y)
    xtest = jnp.linspace(x.min(), x.max(), n_obs)

    # *** MLP Training with EKF ***
    W0 = normal(key_weights, (n_params,)) * 1  # initial random guess
    Q = jnp.eye(n_params) * 1e-4;  # parameters do not change
    R = jnp.eye(1) * sigma_y ** 2;  # observation noise is fixed
    Vinit = jnp.eye(n_params) * 100  # vague prior

    ekf = NLDS(fz, fwd_mlp, Q, R)
    _, ekf_hist = filter(ekf, W0, y[:, None], x[:, None], Vinit, return_params=["mean", "cov"])
    ekf_mu_hist, ekf_Sigma_hist = ekf_hist["mean"], ekf_hist["cov"]

    xtest = jnp.linspace(x.min(), x.max(), 200)
    nframes = n_obs
    fig, ax = plt.subplots()
    camera = Camera(fig)  # initialize camera object

    def func(i):
        W, SW = ekf_mu_hist[i], ekf_Sigma_hist[i]
        W_samples = multivariate_normal(key, W, SW, (100,))
        sample_yhat = fwd_mlp_obs_weights(W_samples, xtest[:, None])
        for sample in sample_yhat:
            ax.plot(xtest, sample, c="tab:gray", alpha=0.07)
        ax.plot(xtest, sample_yhat.mean(axis=0), color='blue')
        ax.scatter(x[:i], y[:i], s=14, c="none", edgecolor="black", label="observations")
        ax.scatter(x[i], y[i], s=30, c="tab:red")
        ax.text(0.4, 1.01, f"EKF+MLP ({i}/{n_obs})", transform=ax.transAxes)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        camera.snap()  # takes snapshot of fig

    for i in range(n_obs):
        func(i)
    ani = camera.animate()  # animates the snapshots
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
