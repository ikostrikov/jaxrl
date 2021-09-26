"""Implementations of algorithms for continuous control."""

import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jaxrl.agents.awac.actor as awr_actor
import jaxrl.agents.sac.critic as sac_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit, static_argnames=('update_target', 'num_samples'))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
        batch: Batch, discount: float, tau: float, num_samples: int,
        beta: float,
        update_target: bool) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = sac_critic.update(key,
                                                actor,
                                                critic,
                                                target_critic,
                                                None,
                                                batch,
                                                discount,
                                                soft_critic=False)
    if update_target:
        new_target_critic = sac_critic.target_update(new_critic, target_critic,
                                                     tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = awr_actor.update(key, actor, new_critic, batch,
                                             num_samples, beta)

    return rng, new_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info
    }


class AWACLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_optim_kwargs: dict = {
                     'learning_rate': 3e-4,
                     'weight_decay': 1e-4
                 },
                 actor_hidden_dims: Sequence[int] = (256, 256, 256, 256),
                 state_dependent_std: bool = False,
                 critic_lr: float = 3e-4,
                 critic_hidden_dims: Sequence[int] = (256, 256),
                 num_samples: int = 1,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 beta: float = 1.0):

        action_dim = actions.shape[-1]

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.num_samples = num_samples
        self.beta = beta

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_def = policies.NormalTanhPolicy(
            actor_hidden_dims,
            action_dim,
            state_dependent_std=state_dependent_std,
            tanh_squash_distribution=False)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adamw(**actor_optim_kwargs))

        critic_def = critic_net.DoubleCritic(critic_hidden_dims)
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
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        new_rng, new_actor, new_critic, new_target_network, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, batch,
            self.discount, self.tau, self.num_samples, self.beta,
            self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_network

        return info
