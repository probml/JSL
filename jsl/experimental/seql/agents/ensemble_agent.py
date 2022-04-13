import jax.numpy as jnp
from jax import jit, value_and_grad, random, vmap, tree_map

import optax

import chex
from flax.core import frozen_dict
import typing_extensions
from typing import Any, NamedTuple

import warnings
from functools import partial

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent

Params = Any
Optimizer = NamedTuple


# https://github.com/deepmind/optax/blob/252d152660300fc7fe22d214c5adbe75ffab0c4a/optax/_src/transform.py#L35
class TraceState(NamedTuple):
    """Holds an aggregation of past updates."""
    trace: chex.ArrayTree


class ModelFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array):
        ...


class LossFn(typing_extensions.Protocol):
    def __call__(self,
                 params: Params,
                 x: chex.Array,
                 y: chex.Array,
                 model_fn: ModelFn) -> float:
        ...


class BeliefState(NamedTuple):
    params: Params
    opt_states: TraceState


class Info(NamedTuple):
    ...


def bootstrap_sampling(key, nsamples):
    def sample(key):
        return random.randint(key, (), 0, nsamples)

    keys = random.split(key, nsamples)
    return vmap(sample)(keys)


def ensemble_agent(classification: bool,
                   loss_fn: LossFn,
                   model_fn: ModelFn,
                   nensembles: int,
                   optimizer: Optimizer = optax.adam(1e-2),
                   buffer_size: int = jnp.inf,
                   nepochs: int = 20,
                   threshold: int = 1,
                   key: chex.PRNGKey = random.PRNGKey(0)):
    assert threshold <= buffer_size

    memory = Memory(buffer_size)
    partial_loss_fn = partial(loss_fn, model_fn=model_fn)
    value_and_grad_fn = jit(value_and_grad(partial_loss_fn))

    def init_state(params: Params):
        opt_states = vmap(optimizer.init)(params)
        return BeliefState(params, opt_states)

    def update(key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):

        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)

        if len(x_) < threshold:
            warnings.warn("There should be more data.", UserWarning)
            info = Info(False, -1, jnp.inf)
            return belief, info

        vbootstrap = vmap(bootstrap_sampling, in_axes=(0, None))

        keys = random.split(key, nensembles)
        indices = vbootstrap(keys, len(x_))

        x_ = jnp.expand_dims(vmap(jnp.take, in_axes=(None, 0))(x_, indices), 2)
        y_ = jnp.expand_dims(vmap(jnp.take, in_axes=(None, 0))(y_, indices), 2)

        def train(params, opt_state, x, y):
            prior = params["params"]["prior"]
            for _ in range(nepochs):
                params = frozen_dict.freeze(
                    {"params": {"prior": prior,
                                "trainable": params["params"]["trainable"]
                                }
                     })
                loss, grads = value_and_grad_fn(params, x, y)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

            params = frozen_dict.freeze(
                {"params": {"prior": prior,
                            "trainable": params["params"]["trainable"]
                            }
                 })
            return params, opt_state

        vtrain = vmap(train, in_axes=(0, 0, 0, 0))
        params, opt_states, = vtrain(belief.params, belief.opt_states, x_, y_)

        return BeliefState(params, opt_states), Info()

    def apply(params: chex.ArrayTree,
              x: chex.Array):
        n = len(x)
        predictions = model_fn(params, x).reshape((n, -1))

        return predictions

    def sample_params(key: chex.PRNGKey,
                      belief: BeliefState):
        sample_key, key = random.split(key)
        index = random.randint(sample_key, (), 0, nensembles)
        params = tree_map(lambda x: x[index], belief.params)
        return params

    return Agent(classification, init_state, update, apply, sample_params)
