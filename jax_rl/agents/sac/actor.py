from typing import Tuple

import jax
import jax.numpy as jnp

from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
