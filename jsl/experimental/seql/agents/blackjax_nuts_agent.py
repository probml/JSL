import chex
import haiku as hk
from jax import jit, random, lax, tree_map, vmap
from jax.tree_util import tree_flatten, tree_unflatten

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

import jax.numpy as jnp

import typing_extensions
from typing import Any, NamedTuple, Union

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
    samples: Any = None


class Info(NamedTuple):
    ...

class NutsState(NamedTuple):
    # https://github.com/blackjax-devs/blackjax/blob/fd83abf6ce16f2c420c76772ff2623a7ee6b1fe5/blackjax/mcmc/integrators.py#L12
    position: chex.ArrayTree
    potential_energy: chex.ArrayTree = None
    potential_energy_grad: chex.ArrayTree = None


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = random.split(rng_key, num_samples)
    final, states = lax.scan(one_step, initial_state, keys)

    return final, states


def blackjax_nuts_agent(key: Union[chex.PRNGKey, int],
                        potential_fn: PotentialFn,
                        model_fn: ModelFn,
                        nsamples: int,
                        nwarmup: int,
                        obs_noise: float = 1.,
                        buffer_size: int = 0):

    rng_key = hk.PRNGSequence(key)

    def init_state(initial_position: Params):
        nuts_state = NutsState(initial_position)
        return BeliefState(nuts_state)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        
        @jit
        def partial_potential_fn(params):
            return potential_fn(params, x, y, model_fn)    

        state = nuts.new_state(belief.state.position,
                                partial_potential_fn)

        kernel_generator = lambda step_size, inverse_mass_matrix: nuts.kernel(partial_potential_fn,
                                                                            step_size,
                                                                            inverse_mass_matrix)
        final_state, (step_size, inverse_mass_matrix), info = stan_warmup.run(next(rng_key),
                                                                            kernel_generator,
                                                                            state,
                                                                            nwarmup)

        # Inference
        nuts_kernel = jit(nuts.kernel(partial_potential_fn,
                                      step_size,
                                      inverse_mass_matrix))

        final, states = inference_loop(next(rng_key),
                                             nuts_kernel,
                                             state,
                                             nsamples)

        belief_state = BeliefState(final,
                                   step_size,
                                   inverse_mass_matrix,
                                   tree_map(lambda x: x[-buffer_size:], states)
                                   )
        return belief_state, Info()
    
    def predict(belief: BeliefState,
                x: chex.Array):

        def get_mean_predictions(samples):

            flat_tree, pytree_def = tree_flatten(samples)
            
            def _predict(*args):
                pytree = tree_unflatten(pytree_def, args[0])
                return model_fn(pytree, x)
            
            return vmap(_predict)(flat_tree)

        predictions = get_mean_predictions(belief.samples.position)
        d, *_ = x.shape
        return jnp.mean(predictions, axis=0), obs_noise * jnp.eye(d)

    return Agent(init_state, update, predict)