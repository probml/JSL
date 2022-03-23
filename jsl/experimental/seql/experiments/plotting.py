
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

def plot_regression_posterior_predictive(ax, env, mu, sigma,
                              model_fn, obs_noise, t):
    #sns.set_style("whitegrid")
    sns.color_palette("pastel")

    nprev = reduce(lambda x, y: x*y,
                   env.X_train[:t].shape[:-1])

    X_train, Y_train = env.X_train[:t+1], env.y_train[:t+1]

    nfeatures = X_train.shape[-1]
    prev_x = X_train.reshape((-1, nfeatures))[:nprev, 1]
    prev_y = Y_train.reshape((-1, 1))[:nprev]

    cur_x = X_train.reshape((-1, nfeatures))[nprev:, 1]
    cur_y = Y_train.reshape((-1, 1))[nprev:]
    
    X_train, Y_train = sort_data(X_train, Y_train)

    posterior_mu, posterior_sigma = posterior_predictive_distribution(X_train,
                                                                    mu,
                                                                    sigma,
                                                                    obs_noise=obs_noise,
                                                                    model_fn=model_fn)

    # Plot training data
    ax.scatter(prev_x, prev_y, alpha=0.5)
    ax.scatter(cur_x, cur_y, alpha=0.5)

    ypred = jnp.squeeze(posterior_mu)
    error = jnp.squeeze(posterior_sigma)

    ax.errorbar(jnp.squeeze(X_train[:, 1]),
                ypred,
                yerr=error,
                color=sns.color_palette()[2])

    ax.fill_between(jnp.squeeze(X_train[:, 1]),
                    ypred + error,
                    ypred - error,
                    alpha=0.2,
                    color=sns.color_palette()[2])

def plot_classification_2d(ax,
                                   env,
                                   mu,
                                   grid,
                                   grid_preds,
                                   t):
    
    sns.set_style("whitegrid")
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)

    X_train, Y_train = sort_data(env.X_train[:t+1], env.y_train[:t+1])
    nclasses = Y_train.max()

    if nclasses == 1 and grid_preds.shape[-1] == 1:
        grid_preds = jnp.hstack([1- grid_preds, grid_preds])

    m = grid_preds
    ax.contourf(grid[:, 1].reshape((100, 100)),
                grid[:, 2].reshape((100, 100)),
                m[:, 1].reshape((100,100)),
                cmap=cmap)
    #ax.plot(mu)
    
    for cls in range(nclasses + 1):
        indices = jnp.argwhere(Y_train == cls)
        
        # Plot training data
        ax.scatter(X_train[indices, 1],
                   X_train[indices, 2])
    
    plt.tight_layout()