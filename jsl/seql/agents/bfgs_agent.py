import typing_extensions
import jax.numpy as jnp
from jax.scipy.optimize import minimize

from typing import NamedTuple

import chex
from jsl.seql.agents.agent import Agent


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
    x: chex.Array


class Info(NamedTuple):
    # True if optimization succeeded
    success: bool
    '''
    0 means converged (nominal)
    1=max BFGS iters reached
    3=zoom failed
    4=saddle point reached
    5=max line search iters reached
    -1=undefined
    '''
    status: int
    # final function value.
    loss: float


def mse(params, inputs, outputs, model_fn):
  predictions = model_fn(params, inputs)
  return jnp.mean(jnp.power(predictions - outputs, 2)) 
  

def bfgs_agent(objective_fn: ObjectiveFn = mse,
               model_fn: ModelFn = lambda mu, x: x @ mu,
               obs_noise: float = 0.01):

    def init_state(x: chex.Array):
        return BeliefState(x)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        
        optimize_results = minimize(objective_fn,
                                    belief.x,
                                    (x, y, model_fn),
                                    method="BFGS")

        info = Info(optimize_results.success,
                    optimize_results.status,
                    optimize_results.fun)
        
        return BeliefState(optimize_results.x), info
    
    def predict(belief: BeliefState,
                x: chex.Array):
        
        d, *_ = x.shape
        return model_fn(belief.x, x), obs_noise * jnp.eye(d)

    return Agent(init_state, update, predict)