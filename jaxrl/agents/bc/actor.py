from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def log_prob_update(actor: Model, batch: Batch,
                    rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -log_probs.mean()
        return actor_loss, {'actor_loss': actor_loss}

    return (rng, *actor.apply_gradient(loss_fn))


def mse_update(actor: Model, batch: Batch,
               rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = actor.apply({'params': actor_params},
                              batch.observations,
                              training=True,
                              rngs={'dropout': key})
        actor_loss = ((actions - batch.actions)**2).mean()
        return actor_loss, {'actor_loss': actor_loss}

    return (rng, *actor.apply_gradient(loss_fn))
