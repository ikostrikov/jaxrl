"""Implementations of algorithms for continuous control."""

import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.ddpg.actor import update as update_actor
from jaxrl.agents.ddpg.critic import update as update_critic
from jaxrl.agents.sac.critic import target_update
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit(
    actor: Model, critic: Model, target_critic: Model, batch: Batch,
    discount: float, tau: float, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    new_critic, critic_info = update_critic(actor, critic, target_critic,
                                            batch, discount)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    new_actor, actor_info = update_actor(actor, new_critic, batch)

    return new_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info,
    }


class DDPGLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 exploration_noise: float = 0.1):
        """
        An implementation of [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
        and Clipped Double Q-Learning (https://arxiv.org/abs/1802.09477).
        """

        action_dim = actions.shape[-1]

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.exploration_noise = exploration_noise

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_def = policies.MSEPolicy(hidden_dims, action_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng,
                                               self.actor.apply_fn,
                                               self.actor.params,
                                               observations,
                                               temperature,
                                               distribution='det')
        self.rng = rng

        actions = np.asarray(actions)
        actions = actions + np.random.normal(
            size=actions.shape) * self.exploration_noise * temperature
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_actor, new_critic, new_target_critic, info = _update_jit(
            self.actor, self.critic, self.target_critic, batch, self.discount,
            self.tau, self.step % self.target_update_period == 0)

        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info
