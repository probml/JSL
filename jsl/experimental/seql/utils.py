import jax.numpy as jnp
from jax import lax, nn, vmap, random

import distrax

import optax

import chex
from typing import NamedTuple, Callable, Optional, Tuple

from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.environments.sequential_data_env import SequentialDataEnvironment
Belief = NamedTuple


def onehot(labels: chex.Array,
           num_classes: int,
           on_value: float = 1.0,
           off_value: float = 0.0) -> chex.Array:
    # https://github.com/google/flax/blob/main/examples/imagenet/train.py
    x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
    x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def binary_cross_entropy(labels: chex.Array,
                         logprobs: chex.Array) -> float:
    probs = jnp.exp(logprobs)
    loss = labels * logprobs + (1 - labels) * jnp.log(1 - probs)
    return -jnp.mean(loss)


def cross_entropy_loss(labels: chex.Array,
                       logprobs: chex.Array,
                       scale: chex.Array = None) -> float:
    nclasses = logprobs.shape[-1]
    if nclasses == 1:
        return binary_cross_entropy(labels, logprobs)
    one_hot_labels = onehot(jnp.squeeze(labels, axis=-1),
                            num_classes=nclasses)
    xentropy = optax.softmax_cross_entropy(logits=logprobs, labels=one_hot_labels)
    return jnp.mean(xentropy)


def categorical_log_likelihood(logprobs: chex.Array,
                               labels: chex.Array) -> float:
    """Computes joint log likelihood based on probs and labels."""
    num_data, nclasses = logprobs.shape
    assert len(labels) == num_data
    one_hot_labels = onehot(labels, num_classes=nclasses)
    assigned_probs = logprobs * one_hot_labels
    return jnp.sum(jnp.log(assigned_probs))


def gaussian_log_likelihood(mu: chex.Array,
                            cov: chex.Array,
                            predictions) -> float:
    return jnp.sum(distrax.MultivariateNormalFullCovariance(jnp.squeeze(mu, axis=-1), cov).log_prob(predictions))


def mse(params: chex.ArrayTree,
        inputs: chex.Array,
        outputs: chex.Array,
        model_fn: Callable) -> float:
    predictions = model_fn(params, inputs)
    return jnp.mean(jnp.power(predictions - outputs, 2))


def posterior_noise(x: chex.Array,
                    sigma: chex.Array,
                    obs_noise: float) -> chex.Array:
    x_ = x.reshape((-1, 1))
    return obs_noise + x_.T.dot(sigma.dot(x_))


def log_lik_joint(key: chex.PRNGKey,
                  agent: Agent,
                  belief: Belief,
                  X: chex.Array,
                  Y: chex.Array,
                  nsamples: int)->float:
    # X: N*J*D, Y: N*J*C  (N=Ntest, J=tau=Njoint)

    def classification_log_lik(theta: chex.ArrayTree,
                               x: chex.Array,
                               y: chex.Array) -> chex.Array:
        logits = nn.log_softmax(agent.apply(params=theta, x=x), axis=-1)
        return distrax.Categorical(logits=logits).log_prob(y)

    def regression_log_lik(theta: chex.ArrayTree,
                           x: chex.Array,
                           y: chex.Array) -> chex.Array:
        mu = agent.apply(params=theta, x=x)
        sigma = agent.get_posterior_cov(belief=belief, x=x)
        return distrax.MultivariateNormalDiag(mu, sigma).log_prob(y)

    def log_lik_per_theta(theta, x, y) -> chex.Array:
        logjoint = lax.cond(agent.is_classifier,
                            vmap(classification_log_lik, in_axes=(None, 0, 0)),
                            vmap(regression_log_lik, in_axes=(None, 0, 0)),
                            theta, x, y)
        return logjoint

    def log_lik_per_sample(key: chex.PRNGKey,
                           belief: Belief,
                           x: chex.Array,
                           y: chex.Array) -> float:
        theta = agent.sample_params(key, belief)
        ll = jnp.sum(vmap(log_lik_per_theta, in_axes=(None, 0, 0))(theta, x, y))
        return ll

    keys = random.split(key, nsamples)
    vlogjoint = vmap(log_lik_per_sample, in_axes=(0, None, 0, 0))
    return jnp.mean(vlogjoint(keys, belief, X, Y), axis=0)


def log_lik(key: chex.PRNGKey,
            agent: Agent,
            belief: Belief,
            x: chex.Array,
            y: chex.Array,
            nsamples: float) -> float:
    N, D = x.shape
    N, C = y.shape

    if agent.is_classifier:
        logits = agent.predict_probs(key, belief, x, nsamples)
        chex.assert_shape(logits, [N, ])
        return distrax.Categorical(logits=logits).log_prob(y).mean(axis=0)
    else:
        mu, sigma = agent.predict_gauss(key, belief, x, nsamples)
        chex.assert_shape(mu, [N, C])
        # chex.assert_shape(sigma, [N, N])
        return distrax.MultivariateNormalDiag(mu, sigma).log_prob(y).mean(axis=0)


# Main function
def train(key: chex.PRNGKey,
          initial_belief_state: Belief,
          agent: Agent,
          env: SequentialDataEnvironment,
          nsteps: int,
          nsamples: int,
          njoint: int,
          callback: Optional[Callable] = None)->Tuple[Belief, chex.Array]:
    rewards = []
    belief = initial_belief_state
    keys = random.split(key, nsteps)

    for t, rng_key in enumerate(keys):
        X_train, Y_train, X_test, Y_test = env.get_data(t)

        data_key, update_key, ll_key, joint_key = random.split(rng_key, 4)
        (X_joint, Y_joint), ll = env.get_joint_data(data_key, nsamples, njoint)

        belief, info = agent.update(update_key,
                                    belief,
                                    X_train,
                                    Y_train)

        if callback:
            if not isinstance(callback, list):
                callback_list = [callback]
            else:
                callback_list = callback

            for f in callback_list:
                f(
                    belief_state=belief,
                    info=info,
                    X_train=X_train,
                    Y_train=Y_train,
                    X_test=X_test,
                    Y_test=Y_test,
                    X_joint=X_joint,
                    Y_joint=Y_joint,
                    true_ll=ll,
                    t=t
                )
        NLL_test = -log_lik(ll_key, agent, belief, X_test, Y_test, nsamples)
        # NLL_joint = -log_lik_joint(joint_key, agent, belief, X_joint, Y_joint, nsamples)

        print(NLL_test)
        # print(NLL_joint)

    return belief, rewards

