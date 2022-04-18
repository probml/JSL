import jax.numpy as jnp
from jax import nn, vmap, random

import chex
from typing import NamedTuple, Tuple, Callable

BeliefState = NamedTuple
Info = NamedTuple
AgentInitFn = Callable
SampleFn = Callable


def gauss_moment_matching(mu: chex.Array) -> Tuple[chex.Array, chex.Array]:
    # m = E[Y] = E_theta[ E[Y|theta] ] ~ 1/S sum_s mu(s)
    # m2 = E[Y^2  ] = E_theta[ E[Y^2| theta] ]  ~ m^2
    # v = V[Y ]  = E[Y^2]  - (E[Y]])^2 ~ m - m^2
    m = jnp.mean(mu, axis=0)
    m2 = jnp.mean(mu ** 2, axis=0)
    v = m - m2
    return m, v


class Agent:
    '''
    Agent interface.
    '''

    def __init__(self,
                 is_classifier: bool):
        self.is_classifier = is_classifier

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array) -> Tuple[BeliefState, Info]:
        pass

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):
        pass

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState) -> chex.Array:
        pass

    def _apply(self,
               params: chex.ArrayTree,
               x: chex.Array):
        n = len(x)
        print("params", params.shape)
        print("x", x.shape)
        predictions = self.model_fn(params, x)
        #predictions = predictions.reshape((n, -1))

        return predictions

    def predict_probs(self,
                      key: chex.PRNGKey,
                      belief: BeliefState,
                      x: chex.Array,
                      nsamples: int) -> chex.Array:
        # p(n, c) = integral_theta Categorical(y=c|xn, theta) p(theta)
        # approx 1/S sum_s Cat(y=c | xn, theta(s))

        def get_probs_per_sample(key: chex.PRNGKey,
                                 x: chex.Array) -> chex.Array:
            theta = self.sample_params(key, belief)

            return nn.softmax(self._apply(params=theta, x=x), axis=-1)

        def get_probs(x: chex.Array) -> chex.Array:
            keys = random.split(key, nsamples)
            probs_per_sample = vmap(get_probs_per_sample, in_axes=(0, None))(keys, x)
            return jnp.mean(probs_per_sample, axis=0)

        probs = vmap(get_probs)(x)

        return probs

    def predict_gauss(self,
                      key: chex.PRNGKey,
                      belief: BeliefState,
                      x: chex.Array,
                      nsamples: int) -> Tuple[chex.Array, chex.Array]:
        # p(y|xn) = integral_theta Gauss(y|mu(xn, theta), sigma(xn,theta)) p(theta)
        # appprox Gauss(y | m_n, v_n)
        # m_n = E[Y|xn] = E_theta[ E[Y|xn, theta] ] ~ 1/S sum_s mu(xn, theta(s))
        # m2_n = E[Y^2 | xn ]  E_theta[ E[Y^2|xn, theta] ]  ~ m_n^2
        # v_n = V[Y|xn ]  = E[Y^2 | xn]]  - (E[Y|xn])^2 ~ m_n - m_n^2
        print("ınit x", x.shape)
        def get_m_and_v_per_sample(key: chex.PRNGKey,
                                   x: chex.Array) -> chex.Array:
            theta = self.sample_params(key, belief)
            m = self._apply(params=theta, x=x)
            print("m", m.shape)
            v = self.get_posterior_cov(belief=belief,
                                       x=x)
            print("v", v.shape)
            return m, v

        def get_m_and_v(x: chex.Array) -> Tuple[chex.Array, chex.Array]:
            keys = random.split(key, nsamples)
            vsample = vmap(get_m_and_v_per_sample, in_axes=(0, None))
            m_per_sample, v_per_sample = vsample(keys, x)
            return gauss_moment_matching(m_per_sample)

        m, v = vmap(get_m_and_v)(x)
        return m, v
