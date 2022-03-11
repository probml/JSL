import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid

import matplotlib.pyplot as plt

# Local imports
from jsl.demos import logreg_biclusters as demo
from jsl.nlds.base import NLDS
from jsl.seql.agents.eekf import eekf
from jsl.seql.environments.sequential_data_env import SequentialDataEnvironment
from jsl.seql.train import train


figures, data = demo.main()

def fz(x): return x
def fx(w, x): return sigmoid(w[None, :] @ x)
def Rt(w, x): return (sigmoid(w @ x) * (1 - sigmoid(w @ x)))[None, None]

def make_biclusters_data_environment(train_batch_size,
                                    test_batch_size):
    
    env = SequentialDataEnvironment(data["Phi"],
                            data["y"].reshape((-1, 1)),
                            data["Phi"],
                            data["y"].reshape((-1, 1)),
                            train_batch_size,
                            test_batch_size,
                            classification=True)
    return env


mean, cov = None, None

def callback_fn(**kwargs):
    global mean, cov

    mu_hist = kwargs["info"].mu_hist
    Sigma_hist = kwargs["info"].Sigma_hist

    if mean is not None:
        mean =jnp.vstack([mean, mu_hist])
        cov =jnp.vstack([cov, Sigma_hist])
    else:
        mean = mu_hist
        cov = Sigma_hist

def main():
    X = data["X"]
    y = data["y"]
    Phi = data["Phi"]
    Xspace = data["Xspace"]
    Phispace = data["Phispace"]
    w_laplace = data["w_laplace"]

    n_datapoints, input_dim = Phi.shape
    colors = ["black" if el else "white" for el in y]

    # Predictive domain
    xmin, ymin = X.min(axis=0) - 0.1
    xmax, ymax = X.max(axis=0) + 0.1
    step = 0.1
    Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
    _, nx, ny = Xspace.shape
    Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])


    train_batch_size = 1
    test_batch_size = 1
    env = make_biclusters_data_environment(train_batch_size,
                                            test_batch_size)
                                            

    ### EEKF Approximation
    mu_t = jnp.zeros(input_dim)
    Pt = jnp.eye(input_dim) * 0.0
    P0 = jnp.eye(input_dim) * 2.0

    nlds = NLDS(fz, fx, Pt, Rt, mu_t, P0)
    agent = eekf(nlds)
    belief = agent.init_state(mu_t, P0)
    unused_rewards = train(belief, agent, env, n_datapoints, callback_fn)

    w_eekf_hist = mean
    P_eekf_hist = cov
    
    w_eekf = mean[-1]
    P_eekf = cov[-1]

 
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
    #P_laplace_diag = jnp.sqrt(jnp.diagonal(SN))
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
    print(w_eekf, end="\n"*2)
    
    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig
    figs = main()
    savefig(figs)
    plt.show()