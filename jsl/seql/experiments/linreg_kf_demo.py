import jax.numpy as jnp

from matplotlib import pyplot as plt
from numpy.linalg import inv

# Local imports
from jsl.seql.environments.base import eveny_spaced_x_sampler
from jsl.seql.environments.sequential_data_env import SequentialDataEnvironment
from jsl.seql.train import train
from jsl.seql.agents.kalman_filter import kalman_filter_reg


def make_matlab_demo_environment(train_batch_size: int= 1,
                                test_batch_size: int=128):
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
    env = make_matlab_demo_environment(test_batch_size=1)

    nsteps, _, input_dim = env.X_train.shape
    
    mu0 = jnp.zeros(input_dim)
    Sigma0 = jnp.eye(input_dim) * 10.

    obs_noise = 1
    agent = kalman_filter_reg(obs_noise)
    belief = agent.init_state(mu0, Sigma0)

    unused_rewards = train(belief, agent, env, nsteps=nsteps, callback=callback_fn)

    w0_hist, w1_hist = mean.T
    w0_err, w1_err = jnp.sqrt(cov[:, [0, 1], [0, 1]].T)

    # Offline estimation
    (w0_post, w1_post), Sigma_post = posterior_lreg(jnp.squeeze(env.X_train),
                                                    jnp.squeeze(env.y_train),
                                                    obs_noise, mu0, Sigma0)

    w0_std, w1_std = jnp.sqrt(Sigma_post[[0, 1], [0, 1]])

    dict_figures = {}

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

if __name__=="__main__":
    main()
    