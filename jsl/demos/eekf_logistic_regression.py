# Online learning of a 2d binary logistic regression model p(y=1|x,w) = sigmoid(w'x),
# using the Exponential-family Extended Kalman Filter (EEKF) algorithm
# described in "Online natural gradient as a Kalman filter", Y. Ollivier, 2018.
# https://projecteuclid.org/euclid.ejs/1537257630.

# The latent state corresponds to the current estimate of the regression weights w.
# The observation model has the form
# p(y(t) |  w(t), x(t)) propto Gauss(y(t) | h_t(w(t)), R(t))
# where h_t(w) = sigmoid(w' * x(t)) = p(t) and  R(t) = p(t) * (1-p(t))

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter

# Import data and baseline solution
from jsl.demos import logreg_biclusters as demo

figures, data = demo.main()
X = data["X"]
y = data["y"]
Phi = data["Phi"]
Xspace = data["Xspace"]
Phispace = data["Phispace"]
w_laplace = data["w_laplace"]


# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)

def sigmoid(x): return jnp.exp(x) / (1 + jnp.exp(x))


def log_sigmoid(z): return z - jnp.log1p(jnp.exp(z))


def fz(x): return x


def fx(w, x): return sigmoid(w[None, :] @ x)


def Rt(w, x): return (sigmoid(w @ x) * (1 - sigmoid(w @ x)))[None, None]


def main():
    N, M = Phi.shape
    n_datapoints, ndims = Phi.shape

    # Predictive domain
    xmin, ymin = X.min(axis=0) - 0.1
    xmax, ymax = X.max(axis=0) + 0.1
    step = 0.1
    Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
    _, nx, ny = Xspace.shape
    Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

    ### EEKF Approximation
    mu_t = jnp.zeros(M)
    Pt = jnp.eye(M) * 0.0
    P0 = jnp.eye(M) * 2.0

    model = NLDS(fz, fx, Pt, Rt)
    (w_eekf, P_eekf), eekf_hist = filter(model, mu_t, y, Phi, P0, return_params=["mean", "cov"])
    w_eekf_hist = eekf_hist["mean"]
    P_eekf_hist = eekf_hist["cov"]

    ### *** Ploting surface predictive distribution ***
    colors = ["black" if el else "white" for el in y]
    dict_figures = {}
    key = random.PRNGKey(31415)
    nsamples = 5000

    # EEKF surface predictive distribution
    eekf_samples = random.multivariate_normal(key, w_eekf, P_eekf, (nsamples,))
    Z_eekf = sigmoid(jnp.einsum("mij,sm->sij", Phispace, eekf_samples))
    Z_eekf = Z_eekf.mean(axis=0)

    fig_eekf, ax = plt.subplots()
    title = "EEKF  Predictive Distribution"
    demo.plot_posterior_predictive(ax, X, Xspace, Z_eekf, title, colors)
    dict_figures["logistic_regression_surface_eekf"] = fig_eekf

    ### Plot EEKF and Laplace training history
    P_eekf_hist_diag = jnp.diagonal(P_eekf_hist, axis1=1, axis2=2)
    # P_laplace_diag = jnp.sqrt(jnp.diagonal(SN))
    lcolors = ["black", "tab:blue", "tab:red"]
    elements = w_eekf_hist.T, P_eekf_hist_diag.T, w_laplace, lcolors
    timesteps = jnp.arange(n_datapoints) + 1

    for k, (wk, Pk, wk_laplace, c) in enumerate(zip(*elements)):
        fig_weight_k, ax = plt.subplots()
        ax.errorbar(timesteps, wk, jnp.sqrt(Pk), c=c, label=f"$w_{k}$ online (EEKF)")
        ax.axhline(y=wk_laplace, c=c, linestyle="dotted", label=f"$w_{k}$ batch (Laplace)", linewidth=3)

        ax.set_xlim(1, n_datapoints)
        ax.legend(framealpha=0.7, loc="upper right")
        ax.set_xlabel("number samples")
        ax.set_ylabel("weights")
        plt.tight_layout()
        dict_figures[f"logistic_regression_hist_ekf_w{k}"] = fig_weight_k

    print("EEKF weights")
    print(w_eekf, end="\n" * 2)

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig

    figs = main()
    savefig(figs)
    plt.show()
