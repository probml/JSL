"""Tests for jsl.sent.agents.laplace_agent"""
import jax.numpy as jnp
from jax import random
from jaxopt import ScipyMinimize

import chex
import itertools

from absl.testing import absltest
from absl.testing import parameterized

from jsl.experimental.seql.agents.laplace_agent import LaplaceAgent
from jsl.experimental.seql.utils import mse


class LaplaceAgentTest(parameterized.TestCase):

    @parameterized.parameters(itertools.product((4,), (1, 0, 5), (0.1,)))
    def test_init_state(self,
                        input_dim: int,
                        buffer_size: int,
                        obs_noise: float):
        output_dim = 1
        model_fn = lambda params, x: x @ params
        solver = ScipyMinimize(fun=mse, method="l-bfgs-b")
        agent = LaplaceAgent(buffer_size=buffer_size,
                             obs_noise=obs_noise,
                             model_fn=model_fn,
                             energy_fn=mse,
                             solver=solver)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim)
        belief = agent.init_state(mu, Sigma)

        chex.assert_shape(belief.mu, mu.shape)
        chex.assert_shape(belief.Sigma, Sigma.shape)

        assert agent.obs_noise == obs_noise
        assert agent.buffer_size == buffer_size

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

        model_fn = lambda params, x: x @ params
        solver = ScipyMinimize(fun=mse, method="l-bfgs-b")
        agent = LaplaceAgent(buffer_size=buffer_size,
                             obs_noise=obs_noise,
                             model_fn=model_fn,
                             energy_fn=mse,
                             solver=solver)
        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim)
        initial_belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, update_key = random.split(key, 4)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        belief, info = agent.update(update_key, initial_belief, x, y)

        chex.assert_shape(belief.mu, (input_dim, output_dim))
        chex.assert_shape(belief.Sigma, (input_dim, input_dim))

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

        model_fn = lambda params, x: x @ params
        solver = ScipyMinimize(fun=mse, method="l-bfgs-b")
        agent = LaplaceAgent(buffer_size=buffer_size,
                             obs_noise=obs_noise,
                             model_fn=model_fn,
                             energy_fn=mse,
                             solver=solver)
        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * obs_noise

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
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

        model_fn = lambda params, x: x @ params
        solver = ScipyMinimize(fun=mse, method="l-bfgs-b")
        agent = LaplaceAgent(buffer_size=buffer_size,
                             obs_noise=obs_noise,
                             model_fn=model_fn,
                             energy_fn=mse,
                             solver=solver)
        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * obs_noise

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, ppd_key = random.split(key)

        x = random.normal(x_key, shape=(ntrain, input_dim))
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

        model_fn = lambda params, x: x @ params
        solver = ScipyMinimize(fun=mse, method="l-bfgs-b")

        agent = LaplaceAgent(buffer_size=buffer_size,
                             obs_noise=obs_noise,
                             model_fn=model_fn,
                             energy_fn=mse,
                             solver=solver)

        mu = jnp.zeros((input_dim, output_dim))
        Sigma = jnp.eye(input_dim) * obs_noise

        belief = agent.init_state(mu, Sigma)

        key = random.PRNGKey(seed)
        x_key, w_key, noise_key, logprob_key = random.split(key, 4)

        x = random.normal(x_key, shape=(ntrain, input_dim))
        w = random.normal(w_key, shape=(input_dim, output_dim))
        y = x @ w + random.normal(noise_key, (ntrain, output_dim))

        samples = agent.logprob_given_belief(logprob_key, belief, x, y, nsamples_params)
        chex.assert_shape(samples, (ntrain, output_dim))
        assert jnp.any(jnp.isinf(samples)) == False
        assert jnp.any(jnp.isnan(samples)) == False

    if __name__ == '__main__':
        absltest.main()
