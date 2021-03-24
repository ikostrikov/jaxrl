"""Implementations of algorithms for continuous control."""

from typing import Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np

import jax_rl.agents.awac.actor as awr_actor
import jax_rl.agents.sac.critic as sac_critic
from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.datasets import Batch
from jax_rl.networks import critic_net, policies
from jax_rl.networks.common import InfoDict, create_model


@jax.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _update_jit(models: ActorCriticTemp, batch: Batch, discount: float,
                tau: float, target_update_period: int, num_samples: int,
                beta: float) -> Tuple[ActorCriticTemp, InfoDict]:

    models, critic_info = sac_critic.update(models,
                                            batch,
                                            discount,
                                            soft_critic=False)
    models = sac_critic.target_update(models, tau, target_update_period)

    models, actor_info = awr_actor.update(models, batch, num_samples, beta)

    return models, {**critic_info, **actor_info}


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

        actor = create_model(
            policies.NormalTanhPolicy(actor_hidden_dims,
                                      action_dim,
                                      state_dependent_std=state_dependent_std),
            [actor_key, observations])
        actor = actor.with_optimizer(flax.optim.Adam(**actor_optim_kwargs))

        critic = create_model(critic_net.DoubleCritic(critic_hidden_dims),
                              [critic_key, observations, actions])
        critic = critic.with_optimizer(
            flax.optim.Adam(learning_rate=critic_lr))
        target_critic = create_model(
            critic_net.DoubleCritic(critic_hidden_dims),
            [critic_key, observations, actions])

        self.models = ActorCriticTemp(actor=actor,
                                      critic=critic,
                                      target_critic=target_critic,
                                      temp=None,
                                      rng=rng)

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(
            self.models.rng, self.models.actor.fn,
            self.models.actor.optimizer.target, observations, temperature)

        self.models = self.models.replace(rng=rng)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.models, info = _update_jit(self.models, batch, self.discount,
                                        self.tau, self.target_update_period,
                                        self.num_samples, self.beta)
        return info
