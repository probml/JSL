import jax.numpy as jnp
from matplotlib import pyplot as plt
from numpy.linalg import inv

from jsl.sent.run import train
from jsl.sent.agents.kalman_filter import KalmanFilterReg

from jsl.sent.environments.base import make_matlab_demo_environment


def posterior_lreg(X, y, R, mu0, Sigma0):
    Sn_bayes_inv = inv(Sigma0) + X.T @ X / R
    Sn_bayes = inv(Sn_bayes_inv)
    mn_bayes = Sn_bayes @ (inv(Sigma0) @ mu0 + X.T @ y / R)

    return mn_bayes, Sn_bayes

def main():
    input_dim = 2 
    mu0 = jnp.zeros(input_dim)
    Sigma0 = jnp.eye(input_dim) * 10.
    F = jnp.eye(input_dim)
    Q, R = 0, 1
    print("1")
    agent = KalmanFilterReg(mu0, Sigma0, F, Q, R)
    env = make_matlab_demo_environment(test_batch_size=1)
    nsteps = 21
    params, rewards = train(agent, env, nsteps=nsteps)

    print(params["mean"].shape)
    print(params["cov"].shape)
    w0_hist, w1_hist = params["mean"].T
    w0_err, w1_err = jnp.sqrt(params["cov"][:, [0, 1], [0, 1]].T)

    # Offline estimation
    input_dim, num_train = 2, 21

    (w0_post, w1_post), Sigma_post = posterior_lreg(jnp.squeeze(env.X_train),
                                                    jnp.squeeze(env.y_train),
                                                    R, mu0, Sigma0)

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

if __name__=="__main__":
    main()
    