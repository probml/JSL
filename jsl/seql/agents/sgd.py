import optax

from jax import jit, value_and_grad

import chex
import typing_extensions
from typing import Any, Callable, NamedTuple

from jsl.seql.agents.agent import Agent


Params = Any
Optimizer = NamedTuple


# https://github.com/deepmind/optax/blob/252d152660300fc7fe22d214c5adbe75ffab0c4a/optax/_src/transform.py#L35
class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""
  trace: chex.ArrayTree


class LossFn(typing_extensions.Protocol):
    def __call__(params: Params,
                 x: chex.Array,
                 y: chex.Array) -> float:
        ...


class BeliefState(NamedTuple):
    params: Params
    opt_state: TraceState


class Info(NamedTuple):
    loss: float


def sgd(loss_fn: LossFn,
        model_fn: Callable,
        optimizer: Optimizer = optax.adam(1e-2)):
    
    value_and_grad_fn = jit(value_and_grad(loss_fn))

    def init_state(params: Params):
        opt_state = optimizer.init(params)
        return BeliefState(params, opt_state)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        params = belief.params 
        opt_state = belief.opt_state
        loss, grads = value_and_grad_fn(params, x, y)

        updates, opt_state = optimizer.update(grads, opt_state)

        return BeliefState(updates, opt_state), Info(loss)


    def predict(belief: BeliefState,
            x: chex.Array): 
    
        params = belief.params
        return model_fn(params, x)

    return Agent(init_state, update, predict)