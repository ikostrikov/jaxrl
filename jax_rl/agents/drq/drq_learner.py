"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.agents.drq.augmentations import batched_random_crop
from jax_rl.agents.drq.networks import DrQDoubleCritic, DrQPolicy
from jax_rl.agents.sac import actor, critic, temperature
from jax_rl.datasets import Batch
from jax_rl.networks import policies
from jax_rl.networks.common import InfoDict, Model


@jax.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _update_jit(drq: ActorCriticTemp, batch: Batch, discount: float,
                tau: float, target_entropy: float,
                update_target: bool) -> Tuple[ActorCriticTemp, InfoDict]:

    rng, key = jax.random.split(drq.rng)
    observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    next_observations = batched_random_crop(key, batch.next_observations)

    batch = batch._replace(observations=observations,
                           next_observations=next_observations)
    drq = drq.replace(rng=rng)

    drq, critic_info = critic.update(drq, batch, discount, soft_critic=True)
    if update_target:
        drq = critic.target_update(drq, tau)

    # Use critic conv layers in actor:
    new_actor_params = drq.actor.params.copy(
        add_or_replace={'SharedEncoder': drq.critic.params['SharedEncoder']})
    new_actor = drq.actor.replace(params=new_actor_params)
    drq = drq.replace(actor=new_actor)

    drq, actor_info = actor.update(drq, batch)
    drq, alpha_info = temperature.update(drq, actor_info['entropy'],
                                         target_entropy)

    return drq, {**critic_info, **actor_info, **alpha_info}


class DrQLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 0.1):

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = DrQPolicy(hidden_dims, action_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = DrQDoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.drq = ActorCriticTemp(actor=actor,
                                   critic=critic,
                                   target_critic=target_critic,
                                   temp=temp,
                                   rng=rng)
        self.step = 0

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.drq.rng,
                                               self.drq.actor.apply_fn,
                                               self.drq.actor.params,
                                               observations, temperature)

        self.drq = self.drq.replace(rng=rng)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        self.drq, info = _update_jit(
            self.drq, batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0)
        return info
