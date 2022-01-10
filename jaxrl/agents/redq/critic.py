from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


def update(rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, backup_entropy: bool,
           n: int, m: int) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    rng, key = jax.random.split(rng)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    all_indx = jnp.arange(0, n)
    rng, key = jax.random.split(rng)
    indx = jax.random.choice(key, a=all_indx, shape=(m, ), replace=False)
    params = jax.tree_util.tree_map(lambda param: param[indx],
                                    target_critic.params)
    next_qs = target_critic.apply_fn({'params': params},
                                     batch.next_observations, next_actions)
    next_q = jnp.min(next_qs, axis=0)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply_fn({'params': critic_params}, batch.observations,
                             batch.actions)
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {'critic_loss': critic_loss, 'qs': qs.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
