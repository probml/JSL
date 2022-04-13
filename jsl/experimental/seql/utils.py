import jax.numpy as jnp
from jax import lax, nn, vmap, random

import distrax

import optax

import chex
from typing import NamedTuple

Agent = NamedTuple
Belief = NamedTuple


def onehot(labels: chex.Array,
           num_classes: int,
           on_value: float = 1.0,
           off_value: float = 0.0):
    # https://github.com/google/flax/blob/main/examples/imagenet/train.py
    x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
    x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def binary_cross_entropy(labels: chex.Array,
                         logprobs: chex.Array):
    probs = jnp.exp(logprobs)
    loss = labels * logprobs + (1 - labels) * jnp.log(1 - probs)
    return -jnp.mean(loss)


def cross_entropy_loss(labels: chex.Array,
                       logprobs: chex.Array,
                       scale: chex.Array = None):
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


def mse(params, inputs, outputs, model_fn):
    predictions = model_fn(params, inputs)
    return jnp.mean(jnp.power(predictions - outputs, 2))


def predict(key, agent, bel, X, nsamples):
    # p(n, c) = integral_theta Categorical(y=c|xn, theta) p(theta)
    # approx 1/S sum_s Cat(y=c | xn, theta(s))

    def get_probs_per_sample(key, x):
        theta = agent.sample_params(key, bel)
        return nn.softmax(agent.apply(params=theta, x=x), axis=-1)

    def get_probs(x):
        keys = random.split(key, nsamples)
        probs_per_sample = vmap(get_probs_per_sample, in_axes=(0, None))(keys, x)
        return jnp.mean(probs_per_sample, axis=0)

    probs = vmap(get_probs)(X)

    return probs


def gauss_moment_matching(mu, sigma):
    # m = E[Y] = E_theta[ E[Y|theta] ] ~ 1/S sum_s mu(s)
    # m2 = E[Y^2  ] = E_theta[ E[Y^2| theta] ]  ~ m^2
    # v = V[Y ]  = E[Y^2]  - (E[Y]])^2 ~ m - m^2
    m = jnp.mean(mu, axis=0)
    m2 = jnp.mean(mu ** 2, axis=0)
    v = m - m2
    return m, v


def predict_gauss(key, agent, bel, X, nsamples):
    # p(y|xn) = integral_theta Gauss(y|mu(xn, theta), sigma(xn,theta)) p(theta)
    # appprox Gauss(y | m_n, v_n)
    # m_n = E[Y|xn] = E_theta[ E[Y|xn, theta] ] ~ 1/S sum_s mu(xn, theta(s))
    # m2_n = E[Y^2 | xn ]  E_theta[ E[Y^2|xn, theta] ]  ~ m_n^2
    # v_n = V[Y|xn ]  = E[Y^2 | xn]]  - (E[Y|xn])^2 ~ m_n - m_n^2

    def get_m_and_v_per_sample(key, x):
        theta = agent.sample_params(key, bel)
        return agent.apply(params=theta, x=x)

    def get_m_and_v(x):
        keys = random.split(key, nsamples)
        vsample = vmap(get_m_and_v_per_sample, in_axes=(0, None))
        m_per_sample, v_per_sample = vsample(keys, x)
        return gauss_moment_matching(m_per_sample, v_per_sample)

    m, v = vmap(get_m_and_v)(X)
    return m, v


def posterior_noise(x: chex.Array,
                    sigma: chex.Array,
                    obs_noise: float):
    x_ = x.reshape((-1, 1))
    return obs_noise + x_.T.dot(sigma.dot(x_))


def log_lik_joint(key: chex.PRNGKey,
                  agent: Agent,
                  belief: Belief,
                  X: chex.Array,
                  Y: chex.Array, nsamples):
    # X: N*J*D, Y: N*J*C  (N=Ntest, J=tau=Njoint)

    def classification_log_lik(theta: chex.ArrayTree,
                               x: chex.Array,
                               y: chex.Array):
        logits = nn.log_softmax(agent.apply(params=theta, x=x), axis=-1)
        return distrax.Categorical(logits=logits).log_prob(y)

    def regression_log_lik(theta: chex.ArrayTree,
                           x: chex.Array,
                           y: chex.Array):
        mu, sigma = agent.apply(params=theta, x=x)
        return distrax.MultivariateNormalDiag(mu, sigma).log_prob(y)

    def log_lik_per_theta(theta, x, y):
        logjoint = lax.cond(agent.classsification,
                            vmap(classification_log_lik, in_axes=(None, 0, 0)),
                            vmap(regression_log_lik, in_axes=(None, 0, 0)),
                            theta, x, y)
        return logjoint

    def log_lik_per_sample(key, belief, x, y):
        theta = agent.sample_params(key, belief)
        ll = jnp.sum(vmap(log_lik_per_theta, in_axes=(None, 0, 0))(theta, x, y))
        return ll

    keys = random.split(key, nsamples)
    vlogjoint = vmap(log_lik_per_sample, in_axes=(0, None, 0, 0))
    return jnp.mean(vlogjoint(keys, belief, X, Y), axis=0)


def log_lik(key, bel, X, Y, nsamples):
    N, D = X.shape
    N, C = Y.shape
    X = jnp.reshape(X, (N, 1, D))
    Y = jnp.reshape(Y, (N, 1, C))
    return log_lik_joint(key, bel, X, Y, nsamples)


# Main function
def train(key,
          initial_belief_state,
          agent,
          env,
          nsteps,
          nsamples,
          njoint,
          callback=None):
    # env.reset()
    # agent.reset()
    rewards = []
    belief = initial_belief_state
    keys = random.split(nsteps)
    for t, rng_key in enumerate(keys):
        X_train, Y_train, X_test, Y_test = env.get_data(t)
        (X_joint, Y_joint), ll = env.get_joint_data(rng_key, nsamples, njoint)

        belief, info = agent.update(belief, X_train, Y_train)

        if callback:
            if not isinstance(callback, list):
                callback_list = [callback]
            else:
                callback_list = callback

            for f in callback_list:
                f(agent=agent,
                  env=env,
                  belief_state=belief,
                  info=info,
                  X_train=X_train,
                  Y_train=Y_train,
                  X_test=X_test,
                  Y_test=Y_test,
                  X_joint=X_joint,
                  Y_joint=Y_joint,
                  true_ll=ll,
                  t=t)

        NLL_test = -log_lik(key, agent, belief, X_test, Y_test)
        NLL_joint = -log_lik_joint(key, agent, belief, X_joint, Y_joint)

    return belief, rewards
