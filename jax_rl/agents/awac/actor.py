from typing import Tuple

import jax
import jax.numpy as jnp

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.agents.awac.value import get_value
from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Params


def update(models: ActorCriticTemp, batch: Batch, num_samples: int,
           beta: float) -> Tuple[ActorCriticTemp, InfoDict]:

    models, v1, v2 = get_value(models, batch, num_samples, soft_value=False)
    v = jnp.minimum(v1, v2)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = models.actor.apply({'params': actor_params}, batch.observations)
        lim = 1 - 1e-5
        actions = jnp.clip(batch.actions, -lim, lim)
        log_probs = dist.log_prob(actions)

        q1, q2 = models.critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        a = q - v

        # we could have used exp(a / beta) here but
        # exp(a / beta) is unbiased but high variance,
        # softmax(a / beta) is biased but lower variance.
        # sum() instead of mean(), because it should be multiplied by batch size.
        actor_loss = -(jax.nn.softmax(a / beta) * log_probs).sum()

        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = models.actor.apply_gradient(actor_loss_fn)

    new_models = models.replace(actor=new_actor)

    return new_models, info
