
import jax.numpy as jnp

import seaborn as sns
from matplotlib import pyplot as plt

from jsl.experimental.seql.utils import posterior_predictive_distribution


def sort_data(x, y):
    *_, nfeatures = x.shape
    *_, ntargets = y.shape

    x_ = x.reshape((-1, nfeatures))
    y_ = y.reshape((-1, ntargets))
    
    if nfeatures>1:
        indices = jnp.argsort(x_[:, 1])
    else:
        indices = jnp.argsort(x_[:, 0])

    x_ = x_[indices]
    y_ = y_[indices]

    return x_, y_

def plot_posterior_predictive(env, mu, sigma, obs_noise, timesteps, filename, **kwargs):
    sns.set_style("whitegrid")
    t = kwargs["t"]

    X_test = kwargs["X_test"]
    Y_test = kwargs["Y_test"]

    X_train, Y_train = sort_data(env.X_train, env.y_train)
    X_test, Y_test = sort_data(X_test, Y_test)


    if t in timesteps:
        if "model_fn" in kwargs:
            m, s = posterior_predictive_distribution(X_train[:t+1],
                                                     mu,
                                                     sigma,
                                                     obs_noise=obs_noise,
                                                     model_fn=kwargs["model_fn"])
        else:
            m, s = posterior_predictive_distribution(X_train[:t+1],
                                                    mu,
                                                    sigma,
                                                    obs_noise=obs_noise)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

        # Plot training data
        ax1.scatter(X_train[:t+1, 1],
                    Y_train[:t+1],
                    label='training data',
                    color="tab:orange")
        
        ax1.errorbar(jnp.squeeze(X_train[:t+1, 1]),
                     jnp.squeeze(m),
                     yerr=jnp.squeeze(s),
                     ecolor="tab:red")
        ax1.set_title("Training Data")

        if "model_fn" in kwargs:
            m, s = posterior_predictive_distribution(X_test,
                                                    mu,
                                                    sigma,
                                                    obs_noise=obs_noise,
                                                    model_fn=kwargs["model_fn"])
        else:
            m, s = posterior_predictive_distribution(X_test,
                                                    mu,
                                                    sigma,
                                                    obs_noise=obs_noise)
        # Plot test data
        ax2.scatter(X_test[:, 1],
                    Y_test,
                    label='test data',
                    color="tab:orange")
        ax2.errorbar(jnp.squeeze(X_test[:, 1]),
                     jnp.squeeze(m),
                     yerr=jnp.squeeze(s),
                     ecolor="tab:red")
        ax2.set_title("Test Data")
        fig.suptitle(f"Posterior Predictive Distribution(t={t})")
        plt.tight_layout()
        plt.savefig(f"jsl/experimental/seql/experiments/figures/{filename}_{t}.png")