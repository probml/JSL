import jax.numpy as jnp
from jax import tree_map, vmap, random

import haiku as hk

from sgmcmcjax.samplers import build_sgld_sampler

import chex
from typing import Any, NamedTuple, Callable

import warnings
import typing_extensions
from functools import partial

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


def sgld_agent(key: chex.PRNGKey,
               loglikelihood: LoglikelihoodFn,
               logprior: LogpriorFn,
               model_fn: ModelFn,
               dt: float,
               batch_size: int,
               nsamples: int,
               obs_noise: float,
               nlast: int = 10,
               buffer_size: int = 0,
               threshold: int = 1):
    # TODO
    partial_loglikelihood = partial(loglikelihood,
                                    model_fn=model_fn)

    rng_key = hk.PRNGSequence(key)

    assert threshold <= buffer_size
    memory = Memory(buffer_size)

    def init_state(params: Params):
        return BeliefState(params)

    def update(belief: BeliefState,
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
        samples = sampler(next(rng_key),
                          nsamples,
                          belief.params)

        final = tree_map(lambda x: x.mean(axis=0),
                         samples)
        samples = tree_map(lambda x: x[-buffer_size:],
                           samples)

        return BeliefState(final, samples, sampler), Info

    def predict(belief: BeliefState,
                x: chex.Array):
        def predict_(params):
            return model_fn(params, x)

        nsamples = len(x)
        predictions = vmap(predict_)(belief.samples)
        predictions = jnp.mean(predictions, axis=0).reshape((nsamples, -1))

        return predictions

    def sample_predictive(key: chex.PRNGKey,
                          belief: BeliefState,
                          x: chex.Array,
                          nsamples: int):

        if belief.sampler is None:
            return jnp.repeat(model_fn(belief.params, x), nsamples, axis=0)

        def sample_and_predict(key):
            params = belief.sampler(key,
                                    1,
                                    belief.params)
            return model_fn(params, x)

        keys = random.split(key, nsamples)
        return vmap(sample_and_predict)(keys)

    return Agent(init_state, update, predict, sample_predictive)
