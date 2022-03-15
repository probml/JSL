import jax.numpy as jnp
from jax import random

from matplotlib import pyplot as plt
from jsl.seql.agents.bayesian_lin_reg_agent import bayesian_reg

from jsl.seql.agents.kf_agent import kalman_filter_reg

from jsl.seql.environments.base import make_random_poly_regression_environment
from jsl.seql.utils import posterior_predictive_distribution
from jsl.seql.utils import train

bayes_belief = None
kf_belief = None


def bayes_callback_fn(**kwargs):
    global bayes_belief
    bayes_belief = kwargs["belief_state"]
    plot_ppd(title="bayes", **kwargs)

def kf_callback_fn(**kwargs):
    global kf_belief
    kf_belief = kwargs["belief_state"]
    plot_ppd(title="kf", **kwargs)


dict_figures = {}
timesteps = [5, 10, 15]

def plot_ppd(**kwargs):
    global x, dict_figures, timesteps

    belief = kwargs["belief_state"]
    t = kwargs["t"]

    X_test = jnp.squeeze(kwargs["X_test"])
    y_test = jnp.squeeze(kwargs["Y_test"])
    indices = jnp.argsort(X_test[:, 1])

    X_test = X_test[indices]
    y_test = y_test[indices]

    if t in timesteps:
        m, s = posterior_predictive_distribution(X_test,
                                                belief.mu,
                                                belief.Sigma,
                                                obs_noise=0.01)
        fig, ax = plt.subplots()
        ax.scatter(X_test[:, 1], y_test, s=140,
                facecolors='none', edgecolors='r',
                label='training data')
        ax.errorbar(X_test[:, 1], jnp.squeeze(m), yerr=jnp.squeeze(s))
        ax.set_title(f"Posterior Predictive Distribution(t={t})")

        title = kwargs["title"]
        dict_figures[f"ppd_{title}_{t}"] = fig
        plt.savefig(f"ppd_{title}_{t}.png")


def main():
    key = random.PRNGKey(0)
    degree = 3
    ntrain = 200  # 80% of the data
    ntest = 50  # 20% of the data

    env = make_random_poly_regression_environment(key,
                                                  degree,
                                                  ntrain,
                                                  ntest)

    obs_noise = 0.01
    kf_agent = kalman_filter_reg(obs_noise)

    input_dim = env.X_train.shape[-1]
    mu0 = jnp.zeros((input_dim,))
    Sigma0 = jnp.eye(input_dim)

    belief = kf_agent.init_state(mu0, Sigma0)

    nsteps = 20
    _, unused_rewards = train(belief, kf_agent, env,
                           nsteps=nsteps, callback=kf_callback_fn)

    buffer_size = jnp.inf
    bayes_agent = bayesian_reg(buffer_size, obs_noise)

    belief = bayes_agent.init_state(mu0.reshape((-1, 1)), Sigma0)

    _, unused_rewards = train(belief, bayes_agent, env,
                           nsteps=nsteps, callback=bayes_callback_fn)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

    X_train, y_train = jnp.squeeze(env.X_train)[:, 1], jnp.squeeze(env.y_train)

    # X_test, y_test = jnp.squeeze(env.X_test)[:, 1], jnp.squeeze(env.y_test)


    indices = jnp.argsort(X_train)
    X_train = jnp.sort(X_train)
    y_train = y_train[indices]

    alpha = 0.5
    y_kf, _  = kf_agent.predict(kf_belief, env.X_train)
    y_kf = jnp.squeeze(y_kf)[indices]

    ax1.plot(X_train, y_kf)
    ax1.scatter(X_train, y_train, alpha=alpha)
    ax1.set_title('Kalman Filter')

    y_bayes, _  = kf_agent.predict(bayes_belief, env.X_train)
    y_bayes = jnp.squeeze(y_bayes)[indices]

    ax2.plot(X_train, y_bayes, c="tab:red")
    ax2.scatter(X_train, y_train, c="tab:red", alpha=alpha)
    ax2.set_title('Bayesian Regression')

    fig.suptitle("Polynomial Regression")
    plt.savefig("sd.png")
    dict_figures["kf_vs_bayes_poly"] = fig

if __name__ == "__main__":
    main()