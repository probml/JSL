import jax.numpy as jnp

from jaxopt import ScipyMinimize

import chex
import typing_extensions
from typing import Any, Dict, NamedTuple, Optional
from functools import partial

import warnings

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.utils import mse

Params = Any


class ModelFn(typing_extensions.Protocol):
    def __call__(self,
                 params: chex.Array,
                 inputs: chex.Array):
        ...


class ObjectiveFn(typing_extensions.Protocol):
    def __call__(self,
                 params: chex.Array,
                 inputs: chex.Array,
                 outputs: chex.Array,
                 model_fn: ModelFn):
        ...


class BeliefState(NamedTuple):
    params: Params


class Info(NamedTuple):
    # https://github.com/google/jaxopt/blob/73a7c48e8dbde912cecd37f3d90401e8d87d574e/jaxopt/_src/scipy_wrappers.py#L47
    # True if optimization succeeded
    fun_val: jnp.ndarray = None
    success: bool = False
    '''
    0 means converged (nominal)
    1=max BFGS iters reached
    3=zoom failed
    4=saddle point reached
    5=max line search iters reached
    -1=undefined
    '''
    status: int = -1
    iter_num: int = 0


def lbfgsb_agent(objective_fn: ObjectiveFn = mse,
                 model_fn: ModelFn = lambda mu, x: x @ mu,
                 tol: Optional[float] = None,
                 options: Optional[Dict[str, Any]] = None,
                 obs_noise: float = 0.01,
                 buffer_size: int = jnp.inf,
                 threshold: int = 1):
    partial_objective_fn = partial(objective_fn,
                                   model_fn=model_fn)

    bfgs = ScipyMinimize(fun=partial_objective_fn,
                         method="L-BFGS-B",
                         tol=tol,
                         options=options)
    assert threshold <= buffer_size

    memory = Memory(buffer_size)

    def init_state(x: chex.Array):
        return BeliefState(x)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)

        if len(x_) < threshold:
            warnings.warn("There should be more data.", UserWarning)
            return belief, Info()

        params, info = bfgs.run(belief.params,
                                inputs=x_,
                                outputs=y_)
        return BeliefState(params), info

    def predict(belief: BeliefState,
                x: chex.Array):

        nsamples = len(x)
        predictions = model_fn(belief.params, x)
        predictions = predictions.reshape((nsamples, -1))

        return predictions

    def sample_predictive(key: chex.PRNGKey,
                             belief: BeliefState,
                             x: chex.Array,
                             nsamples: int):
        return jnp.repeat(predict(belief, x), nsamples, axis=0)

    return Agent(init_state, update, predict, sample_predictive)
