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


class BFGSAgent(Agent):

    def __init__(self,
                 objective_fn: ObjectiveFn = mse,
                 model_fn: ModelFn = lambda mu, x: x @ mu,
                 tol: Optional[float] = None,
                 options: Optional[Dict[str, Any]] = None,
                 min_n_samples: int = 1,
                 buffer_size: int = jnp.inf,
                 obs_noise: float = 0.01,
                 is_classifier: bool = False):
        super(BFGSAgent, self).__init__(is_classifier)
        self.partial_objective_fn = partial(objective_fn,
                                            model_fn=model_fn)

        self.bfgs = ScipyMinimize(fun=self.partial_objective_fn,
                                  method="BFGS",
                                  tol=tol,
                                  options=options)
        assert min_n_samples <= buffer_size

        self.memory = Memory(buffer_size)
        self.objective_fn = objective_fn
        self.model_fn = model_fn
        self.tol = tol
        self.options = options
        self.threshold = min_n_samples
        self.buffer_size = buffer_size
        self.obs_noise = obs_noise

    def init_state(self,
                   x: chex.Array):
        return BeliefState(x)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert self.buffer_size >= len(x)
        x_, y_ = self.memory.update(x, y)

        if len(x_) < self.threshold:
            warnings.warn("There should be more data.", UserWarning)
            return belief, Info()

        params, info = self.bfgs.run(belief.params,
                                     inputs=x_,
                                     outputs=y_)
        return BeliefState(params), info

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):
        n = len(x)
        posterior_cov = jnp.eye(n) * self.obs_noise
        chex.assert_shape(posterior_cov, [n, n])
        return posterior_cov

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        return belief.params
