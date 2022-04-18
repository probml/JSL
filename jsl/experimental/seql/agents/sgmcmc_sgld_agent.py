import jax.numpy as jnp
from jax import tree_map, vmap

from sgmcmcjax.samplers import build_sgld_sampler

import chex
from typing import Any, NamedTuple, Callable

import warnings
import typing_extensions

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent

Params = Any
Samples = Any


class LoglikelihoodFn(typing_extensions.Protocol):

    def __call__(self,
                 params: Params,
                 x: chex.Array,
                 y: chex.Array):
        ...


class LogpriorFn(typing_extensions.Protocol):

    def __call__(self,
                 params: Params,
                 x: chex.Array,
                 y: chex.Array):
        ...


class ModelFn(typing_extensions.Protocol):

    def __call__(self,
                 params: Params,
                 x: chex.Array):
        ...


class BeliefState(NamedTuple):
    params: Params
    samples: Samples = None
    sampler: Callable = None


class Info(NamedTuple):
    ...


class SGLDAgent(Agent):

    def __init__(self,
                 loglikelihood: LoglikelihoodFn,
                 logprior: LogpriorFn,
                 model_fn: ModelFn,
                 dt: float,
                 batch_size: int,
                 nsamples: int,
                 nlast: int = 10,
                 buffer_size: int = 0,
                 threshold: int = 1,
                 obs_noise=0.1,
                 is_classifier: bool = False):
        super(SGLDAgent, self).__init__(is_classifier)
        assert threshold <= buffer_size
        self.memory = Memory(buffer_size)
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.nlast = nlast
        self.batch_size = batch_size
        self.nsamples = nsamples
        self.dt = dt
        self.model_fn = model_fn
        self.logprior = logprior
        self.loglikelihood = loglikelihood

    def init_state(self,
                   params: Params):
        return BeliefState(params)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        assert self.buffer_size >= len(x)

        x_, y_ = self.memory.update(x, y)
        if len(x_) < self.threshold:
            warnings.warn("There should be more data.", UserWarning)
            info = Info(False, -1, jnp.inf)
            return belief, info

        batch_size_ = len(x_) if self.batch_size == -1 else self.batch_size

        sampler = build_sgld_sampler(self.dt,
                                     self.loglikelihood,
                                     self.logprior,
                                     (x_, y_),
                                     batch_size_)
        samples = sampler(key,
                          self.nsamples,
                          belief.params)

        final = tree_map(lambda x: x.mean(axis=0),
                         samples)
        samples = tree_map(lambda x: x[-self.buffer_size:],
                           samples)

        return BeliefState(final, samples, sampler), Info

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):

        n = len(x)
        predictions = vmap(self.model_fn, in_axis=(0, None))(belief.samples,
                                                             x)
        posterior_cov = jnp.diag(jnp.power(jnp.std(predictions, axis=0), 2))
        chex.assert_shape(posterior_cov, [n, n])
        return posterior_cov

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):

        if belief.sampler is None:
            return belief.params

        theta = belief.sampler(key,
                               1,
                               belief.params)

        return theta
