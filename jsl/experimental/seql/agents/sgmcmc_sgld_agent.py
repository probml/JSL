import jax.numpy as jnp
from jax import tree_map

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


def sgld_agent(classification: bool,
               loglikelihood: LoglikelihoodFn,
               logprior: LogpriorFn,
               model_fn: ModelFn,
               dt: float,
               batch_size: int,
               nsamples: int,
               nlast: int = 10,
               buffer_size: int = 0,
               threshold: int = 1):
    assert threshold <= buffer_size
    memory = Memory(buffer_size)

    def init_state(params: Params):
        return BeliefState(params)

    def update(key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        assert buffer_size >= len(x)

        x_, y_ = memory.update(x, y)
        if len(x_) < threshold:
            warnings.warn("There should be more data.", UserWarning)
            info = Info(False, -1, jnp.inf)
            return belief, info

        batch_size_ = len(x_) if batch_size == -1 else batch_size

        sampler = build_sgld_sampler(dt,
                                     loglikelihood,
                                     logprior,
                                     (x_, y_),
                                     batch_size_)
        samples = sampler(key,
                          nsamples,
                          belief.params)

        final = tree_map(lambda x: x.mean(axis=0),
                         samples)
        samples = tree_map(lambda x: x[-buffer_size:],
                           samples)

        return BeliefState(final, samples, sampler), Info

    def apply(params: chex.ArrayTree,
              x: chex.Array):

        n = len(x)
        predictions = model_fn(params, x).reshape((n, -1))

        return predictions

    def sample_params(key: chex.PRNGKey,
                      belief: BeliefState):

        if belief.sampler is None:
            return belief.params

        theta = belief.sampler(key,
                               1,
                               belief.params)

        return theta

    return Agent(classification, init_state, update, apply, sample_params)
