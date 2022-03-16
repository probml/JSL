import jax.numpy as jnp

import chex

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Memory:
    buffer_size: int
    x: chex.Array = None
    y: chex.Array = None

    def update(self,
               x: chex.Array,
               y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        
        if self.x is None or self.buffer_size == len(x):
            new_x, new_y = x, y
        else:
            n = len(x) + len(self.x)

            if self.buffer_size < n:
                nprev = self.buffer_size - len(x)
                new_x = jnp.vstack([self.x[-nprev:], x])
                new_y = jnp.vstack([self.y[-nprev:], y])
            else:
                new_x = jnp.vstack([self.x, x])
                new_y = jnp.vstack([self.y, y])

        self.x = new_x
        self.y = new_y

        return new_x, new_y