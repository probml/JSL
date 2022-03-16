import jax.numpy as jnp
from jax import vmap
from jax.scipy.optimize import minimize

import chex
import typing_extensions
from typing import NamedTuple
from jsl.seql.agents.agent_utils import Memory

from jsl.seql.agents.base import Agent
from jsl.seql.utils import posterior_noise, mse

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
  

def bfgs_agent(objective_fn: ObjectiveFn = mse,
               model_fn: ModelFn = lambda mu, x: x @ mu,
               obs_noise: float = 0.01,
               buffer_size: int = jnp.inf):

    memory = Memory(buffer_size)
    
    def init_state(x: chex.Array):
        return BeliefState(x)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        
        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)

        optimize_results = minimize(objective_fn,
                                    belief.x,
                                    (x_, y_, model_fn),
                                    method="BFGS")

        info = Info(optimize_results.success,
                    optimize_results.status,
                    optimize_results.fun)
        
        return BeliefState(optimize_results.x), info
    
    def predict(belief: BeliefState,
                x: chex.Array):
        try:
            v_posterior_noise = vmap(posterior_noise, in_axes=(0, None, None))
            noise = v_posterior_noise(x, belief.Sigma, obs_noise)
        except:
            d, *_ = x.shape
            noise = obs_noise * jnp.eye(d)
        return model_fn(belief.x, x), noise

    return Agent(init_state, update, predict)