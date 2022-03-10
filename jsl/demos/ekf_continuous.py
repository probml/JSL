# Example of an Extended Kalman Filter using
# a figure-8 nonlinear dynamical system.
# For futher reference and examples see:
#   * Section on EKFs in PML vol2 book
#   * https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb
#   * Nonlinear Dynamics and Chaos - Steven Strogatz

from jsl.demos import plot_utils
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

from jsl.nlds.base import NLDS
from jsl.nlds.continuous_extended_kalman_filter import estimate


def fz(x):
    x, y = x
    return jnp.asarray([y, x - x ** 3])


def fx(x):
    x, y = x
    return jnp.asarray([x, y])


def main():
    dt = 0.01
    T = 7
    nsamples = 70
    x0 = jnp.array([0.5, -0.75])

    # State noise
    Qt = jnp.eye(2) * 0.001
    # Observed noise
    Rt = jnp.eye(2) * 0.01

    key = random.PRNGKey(314)
    ekf = NLDS(fz, fx, Qt, Rt)
    sample_state, sample_obs, jump = ekf.sample(key, x0, T, nsamples)
    mu_hist, V_hist = estimate(ekf, sample_state, sample_obs, jump, dt)

    vmin, vmax, step = -1.5, 1.5 + 0.5, 0.5
    X = np.mgrid[-1:1.5:step, vmin:vmax:step][::-1]
    X_dot = jnp.apply_along_axis(fz, 0, X)

    dict_figures = {}

    fig, ax = plt.subplots()
    ax.plot(*sample_state.T, label="state space")
    ax.scatter(*sample_obs.T, marker="+", c="tab:green", s=60, label="observations")
    field = ax.streamplot(*X, *X_dot, density=1.1, color="#ccccccaa")
    ax.legend()
    plt.axis("equal")
    ax.set_title("State Space")
    dict_figures["ekf-state-space"] = fig

    fig, ax = plt.subplots()
    ax.plot(*sample_state.T, c="tab:orange", label="EKF estimation")
    ax.scatter(*sample_obs.T, marker="+", s=60, c="tab:green", label="observations")
    ax.scatter(*mu_hist[0], c="black", zorder=3)
    for mut, Vt in zip(mu_hist[::4], V_hist[::4]):
        plot_utils.plot_ellipse(Vt, mut, ax, plot_center=False, alpha=0.9, zorder=3)
    plt.legend()
    field = ax.streamplot(*X, *X_dot, density=1.1, color="#ccccccaa")
    ax.legend()
    plt.axis("equal")
    ax.set_title("Approximate Space")
    dict_figures["ekf-estimated-space"] = fig

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    dict_figures = main()
    savefig(dict_figures)
    plt.show()
