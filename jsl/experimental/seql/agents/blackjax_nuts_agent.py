import chex
import haiku as hk
from jax import jit, random, lax

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

import jax.numpy as jnp

import typing_extensions
from typing import Any, NamedTuple, Union
from functools import partial

from jsl.experimental.seql.agents.base import Agent
from jsl.experimental.seql.agents.bfgs_agent import ModelFn


Params = Any
Samples = Any
State = NamedTuple


class ModelFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array):
        ...


class PotentialFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array,
                 y: chex.Array,
                 model_fn: ModelFn):
        ...


class BeliefState(NamedTuple):
    state: State = None
    step_size: float = 0.
    inverse_mass_matrix: chex.Array = None


class Info(NamedTuple):
    samples: Samples


class NutsState(NamedTuple):
    posi
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = random.split(rng_key, num_samples)
    final, states = lax.scan(one_step, initial_state, keys)

    return final, states


def blackjax_nuts_agent(key: Union[chex.PRNGKey, int],
                        model_fn: ModelFn,
                        potential_fn: PotentialFn,
                        nsamples: int,
                        nwarmup: int,
                        obs_noise: float = 1.,
                        buffer_size: int = 0):

    rng_key = hk.PRNGSequence(key)
    kernel_generator = lambda step_size, inverse_mass_matrix: nuts.kernel(potential_fn,
                                                                          step_size,
                                                                          inverse_mass_matrix)
    

    
    def init_state(initial_position: Params):
        nuts_state = NutsState(initial_position)
        return BeliefState(nuts_state)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        
        partial_potential_fn = partial(potential_fn,
                                       x=x, y=y,
                                       model_fn=model_fn)
                                       
        if belief.inverse_mass_matrix is None:
            state = nuts.new_state(belief.state.position,
                                   partial_potential_fn)

            final_state, (step_size, inverse_mass_matrix), info = stan_warmup.run(next(key),
                                                                                kernel_generator,
                                                                                state,
                                                                                nwarmup)

            belief_state = BeliefState(final_state, step_size, inverse_mass_matrix)
        else:
            belief_state = belief

        # Inference
        nuts_kernel = jit(nuts.kernel(partial_potential_fn,
                                      belief_state.step_size,
                                      belief_state.inverse_mass_matrix))

        _, states = inference_loop(next(rng_key),
                                             nuts_kernel,
                                             belief_state.state,
                                             nsamples)

        belief_state = BeliefState(states[-buffer_size:],
                                   belief_state.step_size,
                                   belief_state.inverse_mass_matrix)

        return belief_state, Info()
    
    def predict(belief: BeliefState,
                x: chex.Array):
        params = belief.state.position
        d, *_ = x.shape
        return model_fn(params, x), obs_noise * jnp.eye(d)

    return Agent(init_state, update, predict)