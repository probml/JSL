from typing import Callable
import jax.numpy as jnp
from jax import vmap, lax
from jax.scipy.stats import multivariate_normal

import optax

import chex

def onehot(labels: chex.Array,
           num_classes: int,
           on_value: float =1.0,
           off_value: float=0.0):
  # https://github.com/google/flax/blob/main/examples/imagenet/train.py
  x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
  x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)
  


def binary_cross_entropy(labels, logprobs):
    probs = jnp.exp(logprobs)
    loss = labels * logprobs + (1-labels) * jnp.log(1 - probs)
    return -jnp.mean(loss)

def classification_loss(labels: chex.Array,
                        logprobs: chex.Array,
                        scale: chex.Array = None):
  nclasses = logprobs.shape[-1]
  if nclasses==1:
      return binary_cross_entropy(labels, logprobs)
  one_hot_labels = onehot(labels, num_classes=nclasses)
  xentropy = optax.softmax_cross_entropy(logits=logprobs, labels=one_hot_labels)
  return jnp.mean(xentropy)

def regression_loss(targets:chex.Array,
                    loc: chex.Array,
                    scale: chex.Array):
  # return jnp.mean(jnp.power(predictions - outputs, 2))
  ll = multivariate_normal.logpdf(targets,
                                  jnp.squeeze(loc),
                                  scale,
                                  allow_singular=None)
  return -jnp.mean(ll)


def mse(params, inputs, outputs, model_fn):
  predictions = model_fn(params, inputs)
  return jnp.mean(jnp.power(predictions - outputs, 2)) 


def posterior_noise(x: chex.Array,
                    sigma: chex.Array,
                    obs_noise: float):
    x_ = x.reshape((-1, 1))
    return jnp.sqrt(obs_noise + x_.T.dot(sigma.dot(x_)))


def posterior_predictive_distribution(X: chex.Array,
                                      mu: chex.Array,
                                      sigma: chex.Array,
                                      obs_noise: float,
                                      model_fn: Callable= lambda w, x: x @ w):

    #Determine coefficient distribution
    ppd_mean = model_fn(mu, X)
    v_posterior_noise = vmap(posterior_noise, in_axes=(0, None, None))
    if sigma is None:
        noise = obs_noise
    else:
        noise = v_posterior_noise(X, sigma, obs_noise)
    return ppd_mean, noise


# Main function
def train(initial_belief_state, agent, env, nsteps, callback=None):
    #env.reset()
    #agent.reset()
    rewards = []
    belief_state = initial_belief_state

    for t in range(nsteps):
        X_train, Y_train, X_test, Y_test = env.get_data(t)

        belief_state, info = agent.update(belief_state, X_train, Y_train)
        
        preds = agent.predict(belief_state, X_test)
        reward = env.reward(*preds, Y_test)

        if callback:
            if not isinstance(callback, list):
                callback_list = [callback]
            else:
                callback_list = callback

            for f in callback_list:
                f(belief_state=belief_state,
                  info=info,
                  X_train=X_train,
                  Y_train=Y_train,
                  X_test=X_test,
                  Y_test=Y_test,
                  preds=preds,
                  reward=reward,
                  t=t)
        print(f"Time {t + 1}, Reward: {reward}")
        rewards.append(reward)
        
    return belief_state, rewards