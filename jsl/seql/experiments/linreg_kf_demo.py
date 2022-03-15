import jax.numpy as jnp

from matplotlib import pyplot as plt
from numpy.linalg import inv

# Local imports
from jsl.seql.environments.base import eveny_spaced_x_sampler
from jsl.seql.environments.sequential_data_env import SequentialDataEnvironment
from jsl.seql.utils import posterior_predictive_distribution
from jsl.seql.utils import train
from jsl.seql.agents.kf_agent import kalman_filter_reg


def make_matlab_demo_environment(train_batch_size: int = 1,
                                 test_batch_size: int = 128):
    # Data from original matlab example
    # https://github.com/probml/pmtk3/blob/master/demos/linregOnlineDemoKalman.m

    max_val, N = 20., 21
    X = eveny_spaced_x_sampler(max_val, N)
    Y = jnp.array([2.4865, -0.3033, -4.0531, -4.3359,
                   -6.1742, -5.604, -3.5069, -2.3257,
                   -4.6377, -0.2327, -1.9858, 1.0284,
                   -2.264, -0.4508, 1.1672, 6.6524,
                   4.1452, 5.2677, 6.3403, 9.6264, 14.7842]).reshape((-1, 1))

    env = SequentialDataEnvironment(X, Y,
                                    X, Y,
                                    train_batch_size, test_batch_size,
                                    classification=False)

    return env


def posterior_lreg(X, y, R, mu0, Sigma0):
    Sn_bayes_inv = inv(Sigma0) + X.T @ X / R
    Sn_bayes = inv(Sn_bayes_inv)
    mn_bayes = Sn_bayes @ (inv(Sigma0) @ mu0 + X.T @ y / R)

    return mn_bayes, Sn_bayes


mu_hist, sigma_hist = None, None

def save_history(**kwargs):
    global mu_hist, sigma_hist

    info = kwargs["info"]

    if mu_hist is not None:
        mu_hist = jnp.vstack([mu_hist, info.mu_hist])
        sigma_hist = jnp.vstack([sigma_hist, info.Sigma_hist])
    else:
        mu_hist = info.mu_hist
        sigma_hist = info.Sigma_hist

dict_figures = {}
timesteps = [5, 10, 15, 20]

def plot_ppd(**kwargs):
    global x, dict_figures, timesteps

    belief = kwargs["belief_state"]
    t = kwargs["t"]

    X_test = jnp.squeeze(kwargs["X_test"])
    y_test = jnp.squeeze(kwargs["Y_test"])
    if t in timesteps:
        m, s = posterior_predictive_distribution(X_test,
                                                belief.mu,
                                                belief.Sigma,
                                                obs_noise=0.01)
        fig, ax = plt.subplots()
        ax.scatter(X_test[:t+1, 1], y_test[:t+1], s=140,
                facecolors='none', edgecolors='r',
                label='training data')
        
        ax.errorbar(X_test[:, 1], m, yerr=s)
        ax.set_title(f"Posterior Predictive Distribution(t={t})")
        dict_figures[f"ppd_{t}"] = fig
        # plt.savefig(f"ppd_{t}.png")

def main():
    env = make_matlab_demo_environment(test_batch_size=1)

    nsteps, _, input_dim = env.X_train.shape

    mu0 = jnp.zeros(input_dim)
    Sigma0 = jnp.eye(input_dim) * 10.

    obs_noise = 1
    agent = kalman_filter_reg(obs_noise, return_history=True)
    belief = agent.init_state(mu0, Sigma0)

    callback_fns = [save_history, plot_ppd]
    belief, unused_rewards = train(belief, agent, env, nsteps=nsteps, callback=callback_fns)

    w0_hist, w1_hist = mu_hist.T
    w0_err, w1_err = jnp.sqrt(sigma_hist[:, [0, 1], [0, 1]].T)

    # Offline estimation
    (w0_post, w1_post), Sigma_post = posterior_lreg(jnp.squeeze(env.X_train),
                                                    jnp.squeeze(env.y_train),
                                                    obs_noise, mu0, Sigma0)

    w0_std, w1_std = jnp.sqrt(Sigma_post[[0, 1], [0, 1]])

    timesteps = jnp.arange(nsteps)
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
    ax.set_xlim(-0.5, nsteps)
    dict_figures["linreg_online_kalman"] = fig
    return dict_figures


if __name__ == "__main__":
    main()
