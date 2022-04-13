import jax.numpy as jnp
from jax import hessian, tree_map

import distrax

import chex
import typing_extensions
from typing import Any, NamedTuple, Optional
from functools import partial

import warnings

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent

JaxOptSolver = Any
Params = Any
Info = NamedTuple


class BeliefState(NamedTuple):
    mu: Params
    Sigma: Params = None


class Info(NamedTuple):
    ...


class ModelFn(typing_extensions.Protocol):
    def __call__(self,
                 params: chex.Array,
                 inputs: chex.Array):
        ...


class EnergyFn(typing_extensions.Protocol):
    def __call__(self,
                 params: chex.Array,
                 inputs: chex.Array,
                 outputs: chex.Array,
                 model_fn: ModelFn):
        ...


def laplace_agent(classification: bool,
                  solver: JaxOptSolver,
                  energy_fn: EnergyFn,
                  model_fn: ModelFn,
                  obs_noise: float = 0.01,
                  threshold: int = 1,
                  buffer_size: int = 0):
    assert threshold <= buffer_size

    memory = Memory(buffer_size)

    def init_state(mu: chex.Array,
                   Sigma: Optional[chex.Array] = None):
        return BeliefState(mu, Sigma)

    def update(key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)

        if len(x_) < threshold:
            warnings.warn("There should be more data.", UserWarning)
            return belief, None

        params, info = solver.run(belief.mu, inputs=x_, outputs=y_)
        partial_energy_fn = partial(energy_fn,
                                    inputs=x_,
                                    outputs=y_)

        Sigma = hessian(partial_energy_fn)(params)
        return BeliefState(params, tree_map(jnp.squeeze, Sigma)), info

    def apply(params: chex.ArrayTree,
              x: chex.Array):
        n = len(x)
        predictions = model_fn(params, x)
        predictions = predictions.reshape((n, -1))

        return predictions

    def sample_params(key: chex.PRNGKey,
                      belief: BeliefState):
        mu, Sigma = belief.mu, belief.Sigma
        mvn = distrax.MultivariateNormalFullCovariance(mu, Sigma)
        theta = mvn.sample(seed=key, sample_shape=mu.shape)
        return theta

    return Agent(classification, init_state, update, apply, sample_params)
