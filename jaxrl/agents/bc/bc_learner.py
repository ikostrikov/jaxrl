"""Implementations of algorithms for continuous control."""

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.bc import actor
from jaxrl.datasets import Batch
from jaxrl.networks import autoregressive_policy, policies
from jaxrl.networks.common import InfoDict, Model

_log_prob_update_jit = jax.jit(actor.log_prob_update)
_mse_update_jit = jax.jit(actor.mse_update)


class BCLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-3,
                 num_steps: int = int(1e6),
                 hidden_dims: Sequence[int] = (256, 256),
                 distribution: str = 'det'):

        self.distribution = distribution

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        action_dim = actions.shape[-1]
        if distribution == 'det':
            actor_def = policies.MSEPolicy(hidden_dims,
                                           action_dim,
                                           dropout_rate=0.1)
        elif distribution == 'mog':
            actor_def = policies.NormalTanhMixturePolicy(hidden_dims,
                                                         action_dim,
                                                         dropout_rate=0.1)
        else:
            actor_def = autoregressive_policy.MADETanhMixturePolicy(
                hidden_dims, action_dim, dropout_rate=0.1)

        schedule_fn = optax.cosine_decay_schedule(-actor_lr, num_steps)
        optimiser = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(schedule_fn))

        self.actor = Model.create(actor_def,
                                  inputs=[actor_key, observations],
                                  tx=optimiser)
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = policies.sample_actions(self.rng,
                                                    self.actor.apply_fn,
                                                    self.actor.params,
                                                    observations, temperature,
                                                    self.distribution)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self.distribution == 'det':
            self.rng, self.actor, info = _mse_update_jit(
                self.actor, batch, self.rng)
        else:
            self.rng, self.actor, info = _log_prob_update_jit(
                self.actor, batch, self.rng)
        return info
