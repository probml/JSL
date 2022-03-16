'''
Models that are used for experiments.
'''


import flax.linen as nn

import chex


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