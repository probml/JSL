import jax.numpy as jnp
from jax import vmap, random
from jax.nn.initializers import glorot_normal

import chex

import optax

import flax.linen as nn

import matplotlib.pyplot as plt
from functools import partial

# Prior and trainable networks have the same architecture
from jsl.experimental.seql.agents.ensemble_agent import ensemble_agent
from jsl.experimental.seql.environments.sequential_regression_env import SequentialRegressionEnvironment
from jsl.experimental.seql.utils import mse, train


# Prediction function to be resued in Part 3
def get_predictions(model, params, X):
    y = model._apply(params, X)
    return y


class GenericNet(nn.Module):

    @nn.compact
    def __call__(self, x):
        dense = partial(nn.Dense, kernel_init=glorot_normal())
        x = nn.elu(dense(16)(x))
        x = nn.elu(dense(16)(x))
        x = dense(1)(x)
        return x


# Model that combines prior and trainable nets
class Model(nn.Module):
    prior: GenericNet = GenericNet()
    trainable: GenericNet = GenericNet()
    beta: float = 3

    @nn.compact
    def __call__(self, x):
        x1 = self.prior(x)
        x2 = self.trainable(x)
        return self.beta * x1 + x2


def make_sin_environment(key: chex.PRNGKey,
                         ntrain: int,
                         ntest: int,
                         train_batch_size: int = 1,
                         test_batch_size: int = 1,
                         minval: float = 0.0,
                         maxval: float = 0.5,
                         shuffle: bool = False):
    # Generate dataset and grid
    x_key, y_key = random.split(key)
    n = ntrain + ntest
    X = random.uniform(x_key,
                       shape=(n, 1),
                       minval=minval,
                       maxval=maxval)

    # Define function
    def true_model(key, x):
        epsilons = random.normal(key, shape=(3,)) * 0.02
        return (x + 0.3 * jnp.sin(2 * jnp.pi * (x + epsilons[0])) +
                0.3 * jnp.sin(4 * jnp.pi * (x + epsilons[1])) + epsilons[2])

    # Define vectorized version of function
    target_vmap = vmap(true_model, in_axes=(0, 0))

    # Generate target values
    keys = random.split(y_key, n)
    Y = target_vmap(keys, X)

    X_train = X[:ntrain]
    y_train = Y[:ntrain]

    X_test = X[ntrain:]
    y_test = Y[ntrain:]

    if shuffle:
        env_key, key = random.split(key)
    else:
        env_key = None

    env = SequentialRegressionEnvironment(X_train,
                                          y_train,
                                          X_test,
                                          y_test,
                                          true_model,
                                          train_batch_size,
                                          test_batch_size,
                                          obs_noise=0.0,
                                          key=env_key)

    return env


if __name__ == "__main__":
    # Compute prediction values for each net in ensemble

    key = random.PRNGKey(0)

    init_key, env_key, agent_key = random.split(key, 3)

    beta = 3
    nensembles = 9

    model = Model(beta=beta)
    keys = random.split(init_key, nensembles)
    params = vmap(model.init, in_axes=(0, None))(keys, jnp.ones((10, 1)))

    ntrain, ntest = 100, 100
    batch_size = 10

    env = make_sin_environment(env_key,
                               ntrain=ntrain,
                               ntest=ntest,
                               train_batch_size=batch_size,
                               test_batch_size=batch_size)

    learning_rate = 0.03
    optimizer = optax.adam(learning_rate)
    nepochs = 20

    agent = ensemble_agent(mse,
                           model.apply,
                           nensembles=nensembles,
                           optimizer=optimizer,
                           nepochs=ntrain,
                           key=agent_key)

    initial_belief_state = agent.init_state(params)

    nsteps = 10

    belief_state, rewards = train(initial_belief_state,
                                  agent,
                                  env,
                                  nsteps,
                                  callback=None)

    X_train = env.X_train.reshape((ntrain, -1))
    y_train = env.y_train.reshape((ntrain, -1))
    X_grid = jnp.linspace(-5, 5, 1000).reshape(-1, 1)

    vpredict = vmap(get_predictions, in_axes=(None, 0, None))
    predictions = vpredict(model, belief_state.params, X_grid)

    # Plot the results
    fig, axes = plt.figure(nrows=3, ncols=3)  # figsize=[12,6], dpi=200)
    for ax, pred in zip(axes.flatten(), predictions):
        plt.plot(X_train,
                 y_train,
                 'kx',
                 label='Toy data',
                 alpha=0.8)
        plt.plot(X_grid, pred, label='resultant (g)')
        # plt.title('Predictions of the prior network: random function')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.xlim(-0.5, 1.0)
        plt.ylim(-0.6, 1.4)
        plt.legend()

    plt.tight_layout()
    plt.savefig('randomized_priors_single_model.png')

    plt.show()
