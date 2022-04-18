import jax.numpy as jnp
from jax import hessian, tree_map

import distrax

import chex
import typing_extensions
from typing import Any, NamedTuple, Optional
from functools import partial

import warnings

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent

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


class LaplaceAgent(Agent):

    def __init__(self,
                 solver: JaxOptSolver,
                 energy_fn: EnergyFn,
                 model_fn: ModelFn,
                 obs_noise: float = 0.01,
                 threshold: int = 1,
                 buffer_size: int = 0,
                 is_classifier: bool = False):
        super(LaplaceAgent, self).__init__(is_classifier)

        assert threshold <= buffer_size

        self.memory = Memory(buffer_size)
        self.solver = solver
        self.energy_fn = energy_fn
        self.model_fn = model_fn
        self.obs_noise = obs_noise
        self.threshold = threshold
        self.buffer_size = buffer_size

    def init_state(self,
                   mu: chex.Array,
                   Sigma: Optional[chex.Array] = None):
        return BeliefState(mu, Sigma)

    def update(self,
               key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert self.buffer_size >= len(x)
        x_, y_ = self.memory.update(x, y)

        if len(x_) < self.threshold:
            warnings.warn("There should be more data.", UserWarning)
            return belief, None

        params, info = self.solver.run(belief.mu, inputs=x_, outputs=y_)
        partial_energy_fn = partial(self.energy_fn,
                                    inputs=x_,
                                    outputs=y_)

        Sigma = hessian(partial_energy_fn)(params)
        return BeliefState(params, tree_map(jnp.squeeze, Sigma)), info

    def get_posterior_cov(self,
                          belief: BeliefState,
                          x: chex.Array):
        n = len(x)
        posterior_cov = x @ belief.Sigma @ x.T + self.obs_noise * jnp.eye(n)
        chex.assert_shape(posterior_cov, [n, n])
        return posterior_cov

    def sample_params(self,
                      key: chex.PRNGKey,
                      belief: BeliefState):
        mu, Sigma = belief.mu, belief.Sigma
        mvn = distrax.MultivariateNormalFullCovariance(jnp.squeeze(mu, axis=-1),
                                                       Sigma)
        theta = mvn.sample(seed=key, sample_shape=mu.shape)
        return theta