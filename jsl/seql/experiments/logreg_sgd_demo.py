from functools import partial
import jax.numpy as jnp
from jax import random

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons

from jsl.seql.agents.sgd_agent import sgd_agent
from jsl.seql.experiments.experiment_utils import MLP, cross_entropy_loss
from jsl.seql.environments.sequential_data_env import SequentialDataEnvironment
from jsl.seql.utils import train


cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)


def plot_decision_surface(belief, X, y, apply_fn):
    fig, ax = plt.subplots(figsize=(15, 12))
    grid = jnp.mgrid[-3:3:100j, -3:3:100j].reshape((2, -1)).T
    print(grid.shape)
    preds = apply_fn(belief.params, grid)
    ax.contourf(grid[:, 0].reshape(100, 100),
                grid[:, 1].reshape(100, 100),
                preds.mean(axis=-1).reshape((100, 100)),
                cmap=cmap)

    ax.scatter(X[y== 0, 0], X[y == 0, 1], label='Class 0')
    ax.scatter(X[y== 1, 0], X[y == 1, 1], color='r', label='Class 1')
    sns.despine()
    ax.legend()
    plt.savefig("sjd.png")

def make_moons_env(ntrain: int,
                        ntest: int,
                        noise: float =0.0,
                        train_batch_size: int = 1,
                        test_batch_size: int = 1):
    
    N = ntrain + ntest
    X, y = make_moons(N, noise=noise)

    #min_, max_ = X.min(axis=0), X.max(axis=0)
    #X = (X - min_) / (max_ - min_)
    y = y.reshape((-1, 1))
    
    X_train = jnp.array(X[:ntrain])
    y_train = jnp.array(y[:ntrain])

    X_test = jnp.array(X[ntrain:])
    y_test = jnp.array(y[ntrain:])

    env = SequentialDataEnvironment(X_train,
                                    y_train,
                                    X_test,
                                    y_test,
                                    train_batch_size,
                                    test_batch_size,
                                    classification=True)
    return env


def main():
    key = random.PRNGKey(0)
    ntrain, ntest = 80, 20
    noise = 0.2
    nfeatures, nclasses = 2, 2
    env = make_moons_env(ntrain, ntest, noise=noise)
    
    model = MLP(nclasses)
    partial_cross_entropy_loss = partial(cross_entropy_loss,
                                         predict_fn=model.apply)
    agent = sgd_agent(partial_cross_entropy_loss,
                      model.apply,
                      obs_noise=noise)
    batch = jnp.ones((ntrain, nfeatures))
    variables = model.init(key, batch)
    belief = agent.init_state(variables)

    nsteps = 20
    belief, _ = train(belief, agent, env, nsteps)

    plot_decision_surface(belief,
                          jnp.squeeze(env.X_train),
                          jnp.squeeze(env.y_train),
                          model.apply)




if __name__ == "__main__":
    main()

