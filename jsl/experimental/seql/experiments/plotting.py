
import jax.numpy as jnp

import seaborn as sns
from matplotlib import pyplot as plt

from functools import reduce

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
    sns.color_palette("pastel")

    t = kwargs["t"]

    X_test = kwargs["X_test"]
    Y_test = kwargs["Y_test"]

    nprev = reduce(lambda x, y: x*y, env.X_train[:t].shape[:-1])

    X_train, Y_train = sort_data(env.X_train[:t+1], env.y_train[:t+1])
    X_test, Y_test = sort_data(X_test, Y_test)


    if t in timesteps:
        if "model_fn" in kwargs:
            m, s = posterior_predictive_distribution(X_train,
                                                     mu,
                                                     sigma,
                                                     obs_noise=obs_noise,
                                                     model_fn=kwargs["model_fn"])
        else:
            m, s = posterior_predictive_distribution(X_train,
                                                    mu,
                                                    sigma,
                                                    obs_noise=obs_noise)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

        # Plot training data
        prev_x, prev_y = X_train[:nprev, 1], Y_train[:nprev]
        cur_x, cur_y = X_train[nprev:, 1], Y_train[nprev:]
        
        ax1.scatter(prev_x,
                    prev_y,
                    label='previous training data',
                    alpha=0.1)
        
        ax1.scatter(cur_x,
                    cur_y,
                    label='current training data',
                    alpha=0.1)
        
        error = jnp.squeeze(s)

        ypred = jnp.squeeze(m) 

        ax1.plot(jnp.squeeze(X_train[:, 1]),
                 ypred,
                 color=sns.color_palette()[2])

        ax1.fill_between(jnp.squeeze(X_train[:, 1]),
                         ypred + error,
                        ypred - error,
                        alpha=0.2,
                        color=sns.color_palette()[2])

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
                    alpha=0.3)
        
        ypred = jnp.squeeze(m)
        error = jnp.squeeze(s)

        ax2.plot(jnp.squeeze(X_test[:, 1]),
                ypred,
                color=sns.color_palette()[2])

        ax2.fill_between(jnp.squeeze(X_test[:, 1]),
                         ypred + error,
                         ypred - error,
                         alpha=0.2,
                         color=sns.color_palette()[2])

        ax2.set_title("Test Data")
        fig.suptitle(f"Posterior Predictive Distribution(t={t})")
        plt.tight_layout()
        plt.savefig(f"jsl/experimental/seql/experiments/figures/{filename}_{t}.png")

def plot_classification_predictions(env,
                                   mu,
                                   sigma,
                                   obs_noise,
                                   timesteps,
                                   grid,
                                   grid_preds,
                                   filename,
                                   **kwargs):
    
    sns.set_style("whitegrid")
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)


    X_test = kwargs["X_test"]
    Y_test = kwargs["Y_test"]

    t = kwargs["t"]
    X_train, Y_train = sort_data(env.X_train[:t+1], env.y_train[:t+1])
    X_test, Y_test = sort_data(X_test, Y_test)


    if t in timesteps:

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        nclasses = Y_train.max()

        if nclasses == 1 and grid_preds == 1:
            grid_preds = jnp.hstack([1- grid_preds, grid_preds])

        m = jnp.exp(grid_preds[0])
        ax1.contourf(grid[:, 1].reshape((100, 100)),
                     grid[:, 2].reshape((100, 100)),
                     m.reshape((100,100)),
                     cmap=cmap)
        ax1.set_title("Training Data")

        
        
        ax2.contourf(grid[:, 1].reshape((100, 100)),
                     grid[:, 2].reshape((100, 100)),
                     m.reshape((100,100)),
                     cmap=cmap)

        ax2.set_title("Training Data")
        ax2.set_title("Test Data")


        for cls in range(nclasses + 1):
            indices = jnp.argwhere(Y_train == cls)
            
            # Plot training data
            ax1.scatter(X_train[indices, 1],
                        X_train[indices, 2],
                        label='training data')
            
            # Plot test data
            indices = jnp.argwhere(Y_test == cls)
            ax2.scatter(X_test[indices, 1],
                        X_test[indices, 2],
                        label='test data')

        fig.suptitle(f"Posterior Predictive Distribution(t={t})")
        
        plt.tight_layout()
        plt.savefig(f"jsl/experimental/seql/experiments/figures/{filename}_{t}.png")