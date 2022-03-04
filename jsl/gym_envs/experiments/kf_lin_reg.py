import jax.numpy as jnp

import matplotlib.pyplot as plt

from jsl.gym_envs.configs.regression_kf_test import get_config, make_evenly_spaced_x_sampler, sample_fn
from jsl.gym_envs.run import main as train

from numpy.linalg import inv


def posterior_lreg(X, y, R, mu0, Sigma0):
    Sn_bayes_inv = inv(Sigma0) + X.T @ X / R
    Sn_bayes = inv(Sn_bayes_inv)
    mn_bayes = Sn_bayes @ (inv(Sigma0) @ mu0 + X.T @ y / R)

    return mn_bayes, Sn_bayes

def main():
    config = get_config()
    params_per_trial, _ = train(config)

    trial = 0

    w0_hist, w1_hist = params_per_trial[trial]["mu"].T
    w0_err, w1_err = jnp.sqrt(params_per_trial[trial]["sigma"][:, [0, 1], [0, 1]].T)

    # Offline estimation
    input_dim, num_train = 2, 21
    x_generator = make_evenly_spaced_x_sampler(input_dim)
    X, y = sample_fn(None, x_generator, num_train, None)

    R = config.agent.init_kwargs['R']
    mu0 = config.agent.init_kwargs['mu0']
    Sigma0 = config.agent.init_kwargs['Sigma0']
    (w0_post, w1_post), Sigma_post = posterior_lreg(X, y, R, mu0, Sigma0)

    w0_std, w1_std = jnp.sqrt(Sigma_post[[0, 1], [0, 1]])

    dict_figures = {}

    timesteps = jnp.arange(num_train)
    fig, ax = plt.subplots()
    ax.errorbar(timesteps, w0_hist, w0_err, fmt="-o", label="$w_0$", color="black", fillstyle="none")
    ax.errorbar(timesteps, w1_hist, w1_err, fmt="-o", label="$w_1$", color="tab:red")

    ax.axhline(y=w0_post, c="black", label="$w_0$ batch")
    ax.axhline(y=w1_post, c="tab:red", linestyle="--", label="$w_1$ batch")

    ax.fill_between(timesteps, w0_post - w0_std, w0_post + w0_std, color="black", alpha=0.4)
    ax.fill_between(timesteps, w1_post - w1_std, w1_post + w1_std, color="tab:red", alpha=0.4)
    plt.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("weights")
    ax.set_ylim(-8, 4)
    ax.set_xlim(-0.5, num_train)
    dict_figures["linreg_online_kalman"] = fig

    return dict_figures

if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    dict_figures = main()
    savefig(dict_figures)
    plt.show()

