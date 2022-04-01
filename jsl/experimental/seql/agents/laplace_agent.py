from functools import partial
import typing_extensions
import jax.numpy as jnp
from jax import hessian, vmap

import chex
from typing import Any, NamedTuple, Optional


import warnings

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.utils import posterior_noise


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

def laplace_agent(solver: JaxOptSolver,
               energy_fn: EnergyFn,
               model_fn: ModelFn,
               obs_noise: float = 0.01,
               threshold: int = 1,
               buffer_size : int = 0):

    assert threshold <= buffer_size
    
    memory = Memory(buffer_size)
    
    def init_state(mu: chex.Array,
                   Sigma: Optional[chex.Array]= None):
        return BeliefState(mu, Sigma)

    def update(belief: BeliefState,
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
        return BeliefState(params, jnp.squeeze(Sigma)), info
    
    def predict(belief: BeliefState,
                x: chex.Array):
        nsamples, *_ = x.shape
        ppd_mean = model_fn(belief.mu, x)
        v_posterior_noise = vmap(posterior_noise, in_axes=(0, None, None))
        '''noise = v_posterior_noise(x, belief.Sigma, obs_noise)
        
        nsamples, *_ = x.shape
        noise = noise.reshape((nsamples, -1))
        noise = ppd_mean.reshape((nsamples, -1))'''
        return ppd_mean, None

    return Agent(init_state, update, predict)