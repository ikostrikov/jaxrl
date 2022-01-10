"""Implementations of RedQ.
https://arxiv.org/abs/2101.05982
"""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.redq.actor import update as update_actor
from jaxrl.agents.redq.critic import target_update
from jaxrl.agents.redq.critic import update as update_critic
from jaxrl.agents.sac import temperature
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit,
                   static_argnames=('backup_entropy', 'n', 'm',
                                    'update_target', 'update_policy'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, n: int, m: int,
    update_target: bool, update_policy: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy,
                                            n=n,
                                            m=m)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    if update_policy:
        rng, key = jax.random.split(rng)
        new_actor, actor_info = update_actor(key, actor, new_critic, temp,
                                             batch)
        new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                                  target_entropy)
    else:
        new_actor, actor_info = actor, {}
        new_temp, alpha_info = temp, {}

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class REDQLearner(object):

    def __init__(
            self,
            seed: int,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            actor_lr: float = 3e-4,
            critic_lr: float = 3e-4,
            temp_lr: float = 3e-4,
            n: int = 10,  # Number of critics. 
            m: int = 2,  # Nets to use for critic backups.
            policy_update_delay: int = 20,  # See the original implementation.
            hidden_dims: Sequence[int] = (256, 256),
            discount: float = 0.99,
            tau: float = 0.005,
            target_update_period: int = 1,
            target_entropy: Optional[float] = None,
            backup_entropy: bool = True,
            init_temperature: float = 1.0,
            init_mean: Optional[np.ndarray] = None,
            policy_final_fc_init_scale: float = 1.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.n = n
        self.m = m
        self.policy_update_delay = policy_update_delay

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims, num_qs=n)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        self.step = 0

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

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.n,
            self.m,
            update_target=self.step % self.target_update_period == 0,
            update_policy=self.step % self.policy_update_delay == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
