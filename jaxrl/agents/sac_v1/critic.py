from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def update_v(key: PRNGKey, actor: Model, critic: Model, value: Model,
             temp: Model, batch: Batch,
             soft_critic: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.observations)
    actions = dist.sample(seed=key)
    log_probs = dist.log_prob(actions)
    q1, q2 = critic(batch.observations, actions)
    target_v = jnp.minimum(q1, q2)

    if soft_critic:
        target_v -= temp() * log_probs

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply_fn({'params': value_params}, batch.observations)
        value_loss = ((v - target_v)**2).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
