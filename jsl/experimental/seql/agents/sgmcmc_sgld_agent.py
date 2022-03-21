from functools import partial
import jax.numpy as jnp
from jax import tree_map
import haiku as hk
from typing import Any, NamedTuple
import typing_extensions
import chex
from sgmcmcjax.samplers import build_sgld_sampler
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
               buffer_size: int = 0):

    partial_loglikelihood = partial(loglikelihood,
                                    model_fn=model_fn)

    rng_key = hk.PRNGSequence(key)

    def init_state(params: Params):
        return BeliefState(params)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        sampler = build_sgld_sampler(dt,
                                     loglikelihood,
                                     logprior,
                                     (x, y),
                                     batch_size)
        samples = sampler(next(rng_key),
                          nsamples,
                          belief.params)
        
        final = tree_map(lambda x: x[-1],
                         samples)
        samples = tree_map(lambda x: x[-buffer_size:],
                           samples)

        return BeliefState(final, samples), Info
        
    def predict(belief: BeliefState,
                x: chex.Array):
        mu = jnp.mean(belief.samples, axis=0)
        d, *_ = x.shape
        return model_fn(mu, x), obs_noise * jnp.eye(d)

    return Agent(init_state, update, predict)