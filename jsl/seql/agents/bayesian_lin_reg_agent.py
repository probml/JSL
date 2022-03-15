import jax.numpy as jnp
from jax import vmap

import chex

from dataclasses import dataclass
from typing import NamedTuple, Tuple

# Local imports
from jsl.seql.agents.base import Agent
from jsl.seql.agents.kf_agent import BeliefState
from jsl.seql.utils import posterior_noise

class Info(NamedTuple):
    ...


@dataclass
class Memory:
    buffer_size: int
    x: chex.Array = None
    y: chex.Array = None

    def update(self,
               x: chex.Array,
               y: chex.Array) -> Tuple[chex.Array, chex.Array]:

        if self.x is None:
            new_x, new_y = x, y
        else:
            n = len(x) + len(self.x)
            if self.buffer_size < n:
                nprev = n - self.buffer_size
                new_x = jnp.vstack([self.x[-nprev:], x])
                new_y = jnp.vstack([self.y[-nprev:], y])
            else:
                new_x = jnp.vstack([self.x, x])
                new_y = jnp.vstack([self.y, y])

        self.x = new_x
        self.y = new_y

        return new_x, new_y


def bayesian_reg(buffer_size: int, obs_noise: float = 1.):
    memory = Memory(buffer_size)

    def init_state(mu: chex.Array,
                   Sigma: chex.Array):
        return BeliefState(mu, Sigma)

    def update(belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)

        Sigma0_inv = jnp.linalg.inv(belief.Sigma)
        Sigma_inv = Sigma0_inv + x_.T @ x_ / obs_noise
        Sigma = jnp.linalg.inv(Sigma_inv)
        mu = Sigma @ (Sigma0_inv @ belief.mu + x_.T @ y_ / obs_noise)
        return BeliefState(mu, Sigma), Info()

    def predict(belief: BeliefState,
                x: chex.Array):
        v_posterior_noise = vmap(posterior_noise, in_axes=(0, None, None))
        noise = v_posterior_noise(x, belief.Sigma, obs_noise)
        noise = jnp.diag(jnp.squeeze(noise))
        return x @ belief.mu, noise

    return Agent(init_state, update, predict)
