# Compare extended Kalman filter with unscented kalman filter
# on a nonlinear 2d tracking problem
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsl.nlds.extended_kalman_smoother as eks
from jax import random
from jsl.demos import plot_utils
from jsl.nlds.base import NLDS


def plot_data(sample_state, sample_obs):
    fig, ax = plt.subplots()
    ax.plot(*sample_state.T, label="state space")
    ax.scatter(*sample_obs.T, s=60, c="tab:green", marker="+")
    ax.scatter(*sample_state[0], c="black", zorder=3)
    ax.legend()
    ax.set_title("Noisy observations from hidden trajectory")
    plt.axis("equal")
    return fig, ax


def plot_inference(sample_obs, mean_hist, Sigma_hist, label):
    fig, ax = plt.subplots()
    ax.scatter(*sample_obs.T, marker="+", color="tab:green")
    ax.plot(*mean_hist.T, c="tab:orange", label=label)
    ax.scatter(*mean_hist[0], c="black", zorder=3)
    plt.legend()
    collection = [(mut, Vt) for mut, Vt in zip(mean_hist[::4], Sigma_hist[::4])
                  if Vt[0, 0] > 0 and Vt[1, 1] > 0 and abs(Vt[1, 0] - Vt[0, 1]) < 7e-4]
    for mut, Vt in collection:
        plot_utils.plot_ellipse(Vt, mut, ax, plot_center=False, alpha=0.9, zorder=3)
    plt.axis("equal")
    return fig, ax


def main():
    def fz(x, dt): return x + dt * jnp.array([jnp.sin(x[1]), jnp.cos(x[0])])

    def fx(x, *args): return x

    dt = 0.1
    nsteps = 300
    # Initial state vector
    x0 = jnp.array([1.5, 0.0])
    x0 = jnp.array([1.053, 0.145])
    state_size, *_ = x0.shape
    # State noise
    Qt = jnp.eye(state_size) * 0.001
    # Observed noise
    Rt = jnp.eye(2) * 0.05

    key = random.PRNGKey(31415)
    model = NLDS(lambda x: fz(x, dt), fx, Qt, Rt)
    sample_state, sample_obs = model.sample(key, x0, nsteps)

    # _, ekf_hist = ekf_lib.filter(ekf_model, x0, sample_obs, return_params=["mean", "cov"])
    hist = eks.smooth(model, x0, sample_obs, return_params=["mean", "cov"],
                      return_filter_history=True)
    eks_hist = hist["smooth"]
    ekf_hist = hist["filter"]

    eks_mean_hist = eks_hist["mean"]
    eks_Sigma_hist = eks_hist["cov"]

    ekf_mean_hist = ekf_hist["mean"]
    ekf_Sigma_hist = ekf_hist["cov"]

    dict_figures = {}
    # nlds2d_data
    fig_data, ax = plot_data(sample_state, sample_obs)
    dict_figures["nlds2d_data"] = fig_data

    # nlds2d_ekf
    fig_ekf, ax = plot_inference(sample_obs, ekf_mean_hist, ekf_Sigma_hist,
                                 label="filtered")
    ax.set_title("EKF")
    dict_figures["nlds2d_ekf"] = fig_ekf

    # nlds2d_eks
    fig_eks, ax = plot_inference(sample_obs, eks_mean_hist, eks_Sigma_hist,
                                 label="smoothed")
    ax.set_title("EKS")
    dict_figures["nlds2d_eks"] = fig_eks

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    dict_figures = main()
    savefig(dict_figures)
    plt.show()
