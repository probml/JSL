# This demo exemplifies the use of the Kalman Filter
# algorithm when the linear dynamical system induced by the
# matrix A has imaginary eigenvalues

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jsl.demos.plot_utils import plot_ellipse
from jax import random
from jsl.lds.kalman_filter import KalmanFilter

def plot_uncertainty_ellipses(means, covs, ax):
    timesteps = len(means)
    for t in range(timesteps):
        plot_ellipse(covs[t], means[t], ax, plot_center=False, alpha=0.7)

def main():
    dx = 1.1
    timesteps = 20
    key = random.PRNGKey(27182)

    mean_0 = jnp.array([1, 1, 1, 0]).astype(float)
    Sigma_0 = jnp.eye(4)
    A = jnp.array([
        [0.1, 1.1, dx, 0],
        [-1, 1, 0, dx],
        [0, 0, 0.1, 0],
        [0, 0, 0, 0.1]
    ])
    C = jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    Q = jnp.eye(4) * 0.001
    R = jnp.eye(2) * 4

    lds_instance = KalmanFilter(A, C, Q, R, mean_0, Sigma_0, timesteps)
    state_hist, obs_hist = lds_instance.sample(key)

    res = lds_instance.filter(obs_hist)
    mean_hist, Sigma_hist, mean_cond_hist, Sigma_cond_hist = res
    mean_hist_smooth, Sigma_hist_smooth = lds_instance.smooth(mean_hist, Sigma_hist, mean_cond_hist,
                                                                    Sigma_cond_hist)
    
    dict_figures = {}

    fig_spiral_state, ax = plt.subplots()
    ax.plot(*state_hist[:, :2].T, linestyle="--")
    ax.scatter(*obs_hist.T, marker="+", s=60)
    ax.set_title("State space")
    dict_figures["spiral-state"] = fig_spiral_state


    fig_spiral_filtered, ax = plt.subplots()
    ax.plot(*mean_hist[:, :2].T)
    ax.scatter(*obs_hist.T, marker="+", s=60)
    plot_uncertainty_ellipses(mean_hist[:, :2], Sigma_hist[:, :2, :2], ax)
    ax.set_title("Filtered")
    dict_figures["spiral-filtered"] = fig_spiral_filtered

    fig_spiral_smoothed, ax = plt.subplots()
    ax.plot(*mean_hist_smooth[:, :2].T)
    ax.scatter(*obs_hist.T, marker="+", s=60)
    plot_uncertainty_ellipses(mean_hist_smooth[:, :2], Sigma_hist_smooth[:, :2, :2], ax)
    ax.set_title("Smoothed")
    dict_figures["spiral-smoothed"] = fig_spiral_smoothed

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig
    figures = main()
    savefig(figures)
    plt.show()
