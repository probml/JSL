import optax

import jax.numpy as jnp
from jax import jit, value_and_grad

import chex
import typing_extensions
from typing import Any, Callable, NamedTuple

from jsl.seql.agents.agent_utils import Memory
from jsl.seql.agents.base import Agent


Params = Any
Optimizer = NamedTuple


# https://github.com/deepmind/optax/blob/252d152660300fc7fe22d214c5adbe75ffab0c4a/optax/_src/transform.py#L35
class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""
  trace: chex.ArrayTree


class LossFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array,
                 y: chex.Array) -> float:
        ...


class BeliefState(NamedTuple):
    params: Params
    opt_state: TraceState


class Info(NamedTuple):
    loss: float


def sgd_agent(loss_fn: LossFn,
              model_fn: Callable,
              optimizer: Optimizer = optax.adam(1e-2),
              obs_noise: float = 0.01,
              buffer_size: int = jnp.inf):
              
    memory = Memory(buffer_size)
    value_and_grad_fn = jit(value_and_grad(loss_fn))

    def init_state(params: Params):
        opt_state = optimizer.init(params)
        return BeliefState(params, opt_state)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)

        params = belief.params 
        opt_state = belief.opt_state
        loss, grads = value_and_grad_fn(params, x_, y_)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return BeliefState(params, opt_state), Info(loss)


    def predict(belief: BeliefState,
            x: chex.Array): 
        params = belief.params
        mu_pred = model_fn(params, x)
        d, *_ = mu_pred.shape
        sigma_pred = obs_noise * jnp.eye(d)
        return mu_pred, sigma_pred

    return Agent(init_state, update, predict)