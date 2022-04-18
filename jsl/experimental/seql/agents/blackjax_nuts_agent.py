import warnings
import chex
from jax import jit, random, lax, tree_map, vmap
from jax.tree_util import tree_flatten, tree_unflatten

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

import jax.numpy as jnp

import typing_extensions
from typing import Any, NamedTuple, Callable

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent

Params = Any
Samples = Any
State = NamedTuple


class ModelFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array):
        ...


class LogProbFN(typing_extensions.Protocol):
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
    samples: Samples = None


class Info(NamedTuple):
    ...


class NutsState(NamedTuple):
    # https://github.com/blackjax-devs/blackjax/blob/fd83abf6ce16f2c420c76772ff2623a7ee6b1fe5/blackjax/mcmc/integrators.py#L12
    position: chex.ArrayTree
    potential_energy: chex.ArrayTree = None
    potential_energy_grad: chex.ArrayTree = None
    potential_fn: Callable = None


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = random.split(rng_key, num_samples)
    final, states = lax.scan(one_step, initial_state, keys)

    return final, states


class BlackJaxNutsAgent(Agent):

    def __init__(self,
                 logprob_fn: LogProbFN,
                 model_fn: ModelFn,
                 nsamples: int,
                 nwarmup: int,
                 nlast: int = 10,
                 buffer_size: int = 0,
                 threshold: int = 1,
                 obs_noise: float = 0.1,
                 is_classifier: bool = False):

        super(BlackJaxNutsAgent, self).__init__(is_classifier)

        if buffer_size == jnp.inf:
            buffer_size = 0

        assert threshold <= buffer_size or buffer_size == 0
        self.memory = Memory(buffer_size)

        self.logprob_fn = logprob_fn
        self.model_fn = model_fn
        self.nwarmup = nwarmup
        self.nlast = nlast
        self.nsamples = nsamples

    def init_state(self,
                   initial_position: Params):
        nuts_state = NutsState(initial_position)
        return BeliefState(nuts_state)

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

        @jit
        def partial_potential_fn(params):
            return self.logprob_fn(params, x_, y_, self.model_fn)

        warmup_key, sample_key = random.split(key)

        state = nuts.new_state(belief.state.position,
                               partial_potential_fn)

        kernel_generator = lambda step_size, inverse_mass_matrix: nuts.kernel(partial_potential_fn,
                                                                              step_size,
                                                                              inverse_mass_matrix)
        final_state, (step_size, inverse_mass_matrix), info = stan_warmup.run(warmup_key,
                                                                              kernel_generator,
                                                                              state,
                                                                              self.nwarmup)

        # Inference
        nuts_kernel = jit(nuts.kernel(partial_potential_fn,
                                      step_size,
                                      inverse_mass_matrix))

        final, states = inference_loop(sample_key,
                                       nuts_kernel,
                                       state,
                                       self.nsamples)

        belief_state = BeliefState(tree_map(lambda x: jnp.mean(x, axis=0), states),
                                   step_size,
                                   inverse_mass_matrix,
                                   tree_map(lambda x: x[-self.nlast:], states),
                                   partial_potential_fn)
        return belief_state, Info()

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):

        def get_predictions(samples):
            flat_tree, pytree_def = tree_flatten(samples)

            def _predict(*args):
                pytree = tree_unflatten(pytree_def, args[0])

                return self.model_fn(pytree, x)

            return vmap(_predict)(flat_tree)

        predictions = get_predictions(belief.samples.position)
        posterior_cov = jnp.diag(jnp.power(jnp.std(predictions, axis=0),
                                           2)
                                 )
        n = len(x)
        chex.assert_shape(posterior_cov, [n, n])
        return posterior_cov

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):

        if belief.potential_fn is None:
            return belief.state.position

        state = nuts.new_state(belief.state.position,
                               belief.potential_fn)

        # Inference
        nuts_kernel = jit(nuts.kernel(belief.potential_fn,
                                      belief.step_size,
                                      belief.inverse_mass_matrix))

        final, states = inference_loop(key,
                                       nuts_kernel,
                                       state,
                                       1)

        return final.position
