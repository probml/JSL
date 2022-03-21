from functools import partial
import jax.numpy as jnp
from jax import vmap

from jaxopt import LBFGS

import chex
import typing_extensions
from typing import Any, Callable, NamedTuple, Optional, Union

from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.utils import posterior_noise


Params = Any
AutoOrBoolean = Union[str, bool]


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


def lbfgs_agent(objective_fn: ObjectiveFn,
                model_fn: ModelFn = lambda mu, x: x @ mu,
                obs_noise: float = 1.,
                has_aux: bool = False,
                maxiter: int = 500,
                tol: float = 1e-3,
                condition: str = "strong-wolfe",
                maxls: int = 15,
                decrease_factor: float = 0.8,
                increase_factor: float = 1.5,
                buffer_size: int = 10,
                use_gamma: bool = True,
                implicit_diff: bool = True,
                implicit_diff_solve: Optional[Callable] = None,
                jit: AutoOrBoolean = "auto",
                unroll: AutoOrBoolean = "auto",
                verbose: bool = False):
    '''
    https://github.com/google/jaxopt/blob/53b539e6c5cee4c52262ce17d4601839422ffe87/jaxopt/_src/lbfgs.py#L148
    Attributes:
        fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
        has_aux: whether function fun outputs one (False) or more values (True).
        When True it will be assumed by default that fun(...)[0] is the objective.
        maxiter: maximum number of proximal gradient descent iterations.
        tol: tolerance of the stopping criterion.
        maxls: maximum number of iterations to use in the line search.
        decrease_factor: factor by which to decrease the stepsize during line search
        (default: 0.8).
        increase_factor: factor by which to increase the stepsize during line search
        (default: 1.5).
        history_size: size of the memory to use.
        use_gamma: whether to initialize the inverse Hessian approximation with
        gamma * I, see 'Numerical Optimization', equation (7.20).
        implicit_diff: whether to enable implicit diff or autodiff of unrolled
        iterations.
        implicit_diff_solve: the linear system solver to use.
        jit: whether to JIT-compile the optimization loop (default: "auto").
        unroll: whether to unroll the optimization loop (default: "auto").
        verbose: whether to print error on every iteration or not.
        Warning: verbose=True will automatically disable jit.
    Reference:
        Jorge Nocedal and Stephen Wright.
        Numerical Optimization, second edition.
        Algorithm 7.5 (page 179).
    '''


    partial_objective_fn = partial(objective_fn,
                                   model_fn=model_fn)

    lbfgs = LBFGS(partial_objective_fn,
                  has_aux,
                  maxiter,
                  tol,
                  condition,
                  maxls,
                  decrease_factor,
                  increase_factor,
                  buffer_size,
                  use_gamma,
                  implicit_diff,
                  implicit_diff_solve,
                  jit,
                  unroll,
                  verbose)

    def init_state(params: Params):
        return BeliefState(params)
    

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        params, info = lbfgs.run(belief.params,
                                 inputs=x,
                                 outputs=y)
        return BeliefState(params), info
    
    def predict(belief: BeliefState,
                x: chex.Array):
        d, *_ = x.shape
        noise = obs_noise * jnp.eye(d)
        return model_fn(belief.params, x), noise

    return Agent(init_state, update, predict)
