"""Implementations of algorithms for continuous control."""

from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jax_rl.networks.common import MLP


class Critic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1))(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims)(observations, actions)
        critic2 = Critic(self.hidden_dims)(observations, actions)
        return critic1, critic2
