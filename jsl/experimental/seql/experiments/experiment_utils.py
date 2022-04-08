'''
Models and functions that are used for experiments.
'''

import flax.linen as nn
import matplotlib.pyplot as plt

import chex

from jsl.experimental.seql.utils import train


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


def run_experiment(agents,
                   env,
                   initialize_params,
                   train_batch_size,
                   ntrain,
                   nsteps,
                   nrows,
                   ncols,
                   callback_fn,
                   figsize=(56, 48),
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

        train(belief, agent, env(train_batch_size),
              nsteps=nsteps, callback=partial_callback)

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
            train(belief, batch_agent, env(ntrain),
                  nsteps=1, callback=partial_callback)
