"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac_v1.critic import update_q, update_v
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_value: Model, temp: Model, batch: Batch, discount: float,
    tau: float, target_entropy: float, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_critic, critic_info = update_q(critic, target_value, batch, discount)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)

    rng, key = jax.random.split(rng)
    new_value, value_info = update_v(key, new_actor, new_critic, value, temp,
                                     batch, True)

    if update_target:
        new_target_value = target_update(new_value, target_value, tau)
    else:
        new_target_value = target_value

    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_value, new_target_value, new_temp, {
        **critic_info,
        **value_info,
        **actor_info,
        **alpha_info
    }


class SACV1Learner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = critic_net.ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[critic_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_value = Model.create(value_def,
                                    inputs=[critic_key, observations])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_value = target_value
        self.temp = temp
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_value, new_target_value, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_value,
            self.temp, batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_value = new_target_value
        self.temp = new_temp

        return info
