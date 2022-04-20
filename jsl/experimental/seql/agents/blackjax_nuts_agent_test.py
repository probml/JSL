"""Tests for jsl.sent.agents.blackjax_nuts_agent"""
import jax.numpy as jnp
from jax import random

import chex

import itertools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized

from jsl.experimental.seql.agents.blackjax_nuts_agent import BlackJaxNutsAgent


def objective_fn(params: chex.ArrayTree,
                 inputs: chex.Array,
                 outputs: chex.Array,
                 model_fn: Callable) -> float:
    predictions = model_fn(params, inputs)
    return -jnp.mean(jnp.power(predictions - outputs, 2))


class NutsTest(parameterized.TestCase):

    @parameterized.parameters(itertools.product((4,), (20,), (10,), (20,), (0.1,)))
    def test_init_state(self,
                        input_dim: int,
                        nsamples: int,
                        nwarmup: int,
                        buffer_size: int,
                        obs_noise: float):
        output_dim = 1
        model_fn = lambda params, x: x @ params
        agent = BlackJaxNutsAgent(objective_fn,
                                  model_fn,
                                  nsamples,
                                  nwarmup,
                                  buffer_size=buffer_size,
                                  obs_noise=obs_noise)
        params = jnp.zeros((input_dim, output_dim))
        belief = agent.init_state(params)
        chex.assert_shape(belief.state.position, params.shape)

        assert agent.obs_noise == obs_noise
        assert agent.buffer_size == buffer_size
        assert agent.nsamples == nsamples
        assert agent.nwarmup == nwarmup

    @parameterized.parameters(itertools.product((0,),
                                                (10,),
                                                (2,),
                                                (10,),
                                                (0.1,)))
    def test_update(self,
                    seed: int,
                    ntrain: int,
                    input_dim: int,
                    buffer_size: int,
                    obs_noise: float):
        output_dim = 1
        nsamples, nwarmup = 20, 10
        model_fn = lambda params, x: x @ params
        agent = BlackJaxNutsAgent(objective_fn,
                                  model_fn,
                                  nsamples,
                                  nwarmup,
                                  buffer_size=buffer_size,
                                  obs_noise=obs_noise)

        params = jnp.zeros((input_dim, output_dim))
        initial_belief = agent.init_state(params)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, update_key = random.split(key, 4)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        belief, info = agent.update(update_key, initial_belief, x, y)

        chex.assert_shape(belief.state.position, (input_dim, output_dim))

    @parameterized.parameters(itertools.product((0,),
                                                (2,),
                                                (10,),
                                                (0.1,)))
    def test_sample_params(self,
                           seed: int,
                           input_dim: int,
                           buffer_size: int,
                           obs_noise: float):
        output_dim = 1
        nsamples, nwarmup = 20, 10
        model_fn = lambda params, x: x @ params
        agent = BlackJaxNutsAgent(objective_fn,
                                  model_fn,
                                  nsamples,
                                  nwarmup,
                                  buffer_size=buffer_size,
                                  obs_noise=obs_noise)

        params = jnp.zeros((input_dim, output_dim))
        belief = agent.init_state(params)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, update_key, sample_key = random.split(key, 5)

        ntrain = 10
        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        belief, info = agent.update(update_key, belief, x, y)

        theta = agent.sample_params(key, belief)

        chex.assert_shape(theta, (input_dim, output_dim))

    @parameterized.parameters(itertools.product((0,),
                                                (10,),
                                                (2,),
                                                (10,),
                                                (5,),
                                                (10,),
                                                (0.1,)))
    def test_posterior_predictive_sample(self,
                                         seed: int,
                                         ntrain: int,
                                         input_dim: int,
                                         nsamples_params: int,
                                         nsamples_output: int,
                                         buffer_size: int,
                                         obs_noise: float,
                                         ):
        output_dim = 1
        nsamples, nwarmup = 20, 10
        model_fn = lambda params, x: x @ params
        agent = BlackJaxNutsAgent(objective_fn,
                                  model_fn,
                                  nsamples,
                                  nwarmup,
                                  buffer_size=buffer_size,
                                  obs_noise=obs_noise)

        params = jnp.zeros((input_dim, output_dim))
        belief = agent.init_state(params)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, update_key, ppd_key = random.split(key, 5)

        ntrain = 10
        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        belief, info = agent.update(update_key, belief, x, y)

        samples = agent.posterior_predictive_sample(key, belief, x, nsamples_params, nsamples_output)
        chex.assert_shape(samples, (nsamples_params, ntrain, nsamples_output, output_dim))

    @parameterized.parameters(itertools.product((0,),
                                                (5,),
                                                (2,),
                                                (10,),
                                                (10,),
                                                (0.1,)))
    def test_logprob_given_belief(self,
                                  seed: int,
                                  ntrain: int,
                                  input_dim: int,
                                  nsamples_params: int,
                                  buffer_size: int,
                                  obs_noise: float,
                                  ):
        output_dim = 1
        nsamples, nwarmup = 20, 10
        model_fn = lambda params, x: x @ params
        agent = BlackJaxNutsAgent(objective_fn,
                                  model_fn,
                                  nsamples,
                                  nwarmup,
                                  buffer_size=buffer_size,
                                  obs_noise=obs_noise)

        params = jnp.zeros((input_dim, output_dim))

        belief = agent.init_state(params)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, update_key, logprob_key = random.split(key, 5)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        belief, info = agent.update(update_key, belief, x, y)

        samples = agent.logprob_given_belief(logprob_key, belief, x, y, nsamples_params)
        chex.assert_shape(samples, (ntrain, output_dim))
        assert jnp.any(jnp.isinf(samples)) == False
        assert jnp.any(jnp.isnan(samples)) == False

    if __name__ == '__main__':
        absltest.main()
