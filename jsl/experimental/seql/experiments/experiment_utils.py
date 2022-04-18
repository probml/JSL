'''
Models and functions that are used for experiments.
'''
from jax import random

import flax.linen as nn

import matplotlib.pyplot as plt

import chex
from typing import List, Callable, Tuple

from jsl.experimental.seql.utils import train
from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.environments.sequential_data_env import SequentialDataEnvironment

class MLP(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x: chex.Array
                 ):
        x = nn.Dense(50, name="last_layer")(x)
        x = nn.relu(x)
        x = nn.Dense(self.nclasses)(x)
        return x


class LeNet5(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x: chex.Array):
        x = x if len(x.shape) > 1 else x[None, :]
        x = x.reshape((x.shape[0], 28, 28, 1))
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)

        x = nn.avg_pool(x,
                        window_shape=(2, 2),
                        strides=(2, 2),
                        padding="VALID")

        x = nn.Conv(features=16,
                    kernel_size=(5, 5),
                    padding="VALID")(x)

        x = nn.relu(x)
        x = nn.avg_pool(x,
                        window_shape=(2, 2),
                        strides=(2, 2),
                        padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84,
                     name="last_layer")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.nclasses)(x)
        return x.squeeze()


def run_experiment(key: chex.PRNGKey,
                   agents: List[Agent],
                   env: SequentialDataEnvironment,
                   initialize_params: Callable,
                   train_batch_size: int,
                   ntrain: int,
                   nsteps: int,
                   nsamples: int,
                   njoint: int,
                   nrows: int,
                   ncols: int,
                   callback_fn: Callable,
                   figsize: Tuple[int, int] = (56, 48),
                   **init_kwargs):

    batch_agents_included = "batch_agents" in init_kwargs

    if nrows != 1:
        fig, big_axes = plt.subplots(nrows=nrows,
                                     ncols=1,
                                     figsize=figsize)
    else:
        fig, big_axes = plt.subplots(nrows=nrows,
                                     ncols=ncols,
                                     figsize=figsize)
        big_axes = big_axes.flatten()

    for idx, (big_ax, (agent_name, agent)) in enumerate(zip(big_axes, agents.items())):
        big_ax.set_title(agent_name.upper(), fontsize=36, y=1.2)
        if nrows != 1:
            # Turn off axis lines and ticks of the big subplot
            # obs alpha is 0 in RGBA string!
            big_ax.tick_params(labelcolor=(1., 1., 1., 0.0),
                               top='off',
                               bottom='off',
                               left='off',
                               right='off')
            # removes the white frame
            big_ax._frameon = False

        params = initialize_params(agent_name, **init_kwargs)
        belief = agent.init_state(*params)

        partial_callback = lambda **kwargs: callback_fn(agent,
                                                        env(train_batch_size),
                                                        agent_name,
                                                        fig=fig,
                                                        nrows=nrows,
                                                        ncols=ncols,
                                                        idx=idx,
                                                        **init_kwargs,
                                                        **kwargs)

        train_key, key = random.split(key)
        train(train_key,
              belief,
              agent,
              env(train_batch_size),
              nsamples=nsamples,
              njoint=njoint,
              nsteps=nsteps,
              callback=partial_callback)

        if batch_agents_included:
            batch_agent = init_kwargs["batch_agents"][agent_name]
            partial_callback = lambda **kwargs: callback_fn(agent,
                                                            env(ntrain),
                                                            agent_name,
                                                            fig=fig,
                                                            nrows=nrows,
                                                            ncols=ncols,
                                                            idx=idx,
                                                            title="Batch Agent",
                                                            subplot_idx=(idx + 1) * ncols,
                                                            **init_kwargs,
                                                            **kwargs)

            train_key, key = random.split(key)
            train(train_key,
                  belief,
                  batch_agent,
                  env(ntrain),
                  nsamples=nsamples,
                  njoint=njoint,
                  nsteps=1,
                  callback=partial_callback)