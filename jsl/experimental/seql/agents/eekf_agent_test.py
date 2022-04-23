"""Tests for jsl.sent.agents.eekf_agent"""
import jax.numpy as jnp
from jax import random, nn

import chex

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from jsl.experimental.seql.agents.eekf_agent import EEKFAgent
from jsl.nlds.base import NLDS


def fz(x):
    return x


def fx(w, x):
    return (x @ w)[None, ...]


def Rt(w, x):
    return (x @ w * (1 - x @ w))[None, None]


class EEKFAgentTest(parameterized.TestCase):

    @parameterized.parameters((4,))
    def test_init_state(self,
                        input_dim: int):
        output_dim = 1

        Pt = jnp.eye(input_dim) * 0.0
        P0 = jnp.eye(input_dim) * 2.0
        mu0 = jnp.zeros((input_dim,))
        nlds = NLDS(fz, fx, Pt, Rt, mu0, P0)

        agent = EEKFAgent(nlds,
                          is_classifier=True)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim)
        belief = agent.init_state(mu, Sigma)

        chex.assert_shape(belief.mu, mu.shape)
        chex.assert_shape(belief.Sigma, Sigma.shape)


    @parameterized.parameters(itertools.product((0,),
                                                (2,)))
    def test_sample_params(self,
                           seed: int,
                           input_dim: int):
        output_dim = 1

        Pt = jnp.eye(input_dim) * 0.0
        P0 = jnp.eye(input_dim) * 2.0
        mu0 = jnp.zeros((input_dim,))
        nlds = NLDS(fz, fx, Pt, Rt, mu0, P0)

        agent = EEKFAgent(nlds,
                          is_classifier=True)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * 0.2

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        theta = agent.sample_params(key, belief)

        chex.assert_shape(theta, (input_dim, output_dim))

    @parameterized.parameters(itertools.product((0,),
                                                (10,),
                                                (2,),
                                                (10,),
                                                (5,)))
    def test_posterior_predictive_sample(self,
                                         seed: int,
                                         ntrain: int,
                                         input_dim: int,
                                         nsamples_params: int,
                                         nsamples_output: int,
                                         ):
        output_dim = 1

        Pt = jnp.eye(input_dim) * 0.0
        P0 = jnp.eye(input_dim) * 2.0
        mu0 = jnp.zeros((input_dim,))
        nlds = NLDS(fz, fx, Pt, Rt, mu0, P0)

        agent = EEKFAgent(nlds,
                          is_classifier=True)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * 0.2

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, ppd_key = random.split(key)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        samples = agent.posterior_predictive_sample(key, belief, x, nsamples_params, nsamples_output)
        chex.assert_shape(samples, (nsamples_params, ntrain, nsamples_output, output_dim))

    @parameterized.parameters(itertools.product((0,),
                                                (5,),
                                                (3,),
                                                (2,),
                                                (10,)))
    def test_logprob_given_belief(self,
                                  seed: int,
                                  ntrain: int,
                                  input_dim: int,
                                  output_dim: int,
                                  nsamples_params: int
                                  ):


        Pt = jnp.eye(input_dim) * 0.0
        P0 = jnp.eye(input_dim) * 2.0
        mu0 = jnp.zeros((input_dim,))
        nlds = NLDS(fz, fx, Pt, Rt, mu0, P0)

        agent = EEKFAgent(nlds,
                          is_classifier=True)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * 0.2

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, logprob_key = random.split(key, 4)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = nn.softmax(x @ w + random.normal(noise_key, (ntrain, output_dim)), axis=-1)

        samples = agent.logprob_given_belief(logprob_key, belief, x, y, nsamples_params)
        chex.assert_shape(samples, (ntrain, output_dim))

        assert jnp.any(jnp.isinf(samples)) == False
        assert jnp.any(jnp.isnan(samples)) == False

    if __name__ == '__main__':
        absltest.main()
