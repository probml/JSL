import jax.numpy as jnp
from jax import vmap, lax

import optax

import flax.linen as nn

class MLP(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(50, name="last_layer")(x))
        x = nn.Dense(self.nclasses)(x)
        return x

class LeNet5(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x):
        x = x if len(x.shape) > 1 else x[None, :]
        x = x.reshape((x.shape[0], 28, 28, 1))
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84, name="last_layer")(x)  # There are 10 classes in MNIST
        x = nn.relu(x)
        x = nn.Dense(features=self.nclasses)(x)
        return x.squeeze()


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
  x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


def cross_entropy_loss(params, inputs, labels, predict_fn):
  logits = predict_fn(params, inputs)
  nclasses = logits.shape[-1]
  one_hot_labels = onehot(labels, num_classes=nclasses)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def posterior_predictive_distribution(X, mu, sigma, obs_noise):
    #Determine coefficient distribution
    ppd_mean = X @ mu
    def posterior_noise(x):
       return jnp.sqrt(obs_noise + x.T.dot(sigma.dot(x)))

    noise = vmap(posterior_noise)(X)
    return ppd_mean, noise