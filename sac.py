"""Implementations of algorithms for continuous control."""

import copy
import typing

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.optim.base import Optimizer
from tensorflow_probability.substrates import jax as tfp

import rl_types

tfd = tfp.distributions
tfb = tfp.bijectors

PRNGKey = typing.Any
Params = flax.core.frozen_dict.FrozenDict


class MLP(nn.Module):
    hidden_dims: typing.Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        for size in self.hidden_dims[:-1]:
            x = nn.Dense(size,
                         kernel_init=nn.initializers.orthogonal(
                             jnp.sqrt(2.0)))(x)
            x = nn.relu(x)
        x = nn.Dense(self.hidden_dims[-1],
                     kernel_init=nn.initializers.orthogonal(1e-2))(x)

        return x


class DoubleCritic(nn.Module):
    hidden_dims: typing.Sequence[int]

    @nn.compact
    def __call__(
        self, observations: jnp.DeviceArray, actions: jnp.DeviceArray
    ) -> typing.Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        inputs = jnp.concatenate([observations, actions], -1)
        critic1 = MLP((*self.hidden_dims, 1))(inputs)
        critic2 = MLP((*self.hidden_dims, 1))(inputs)
        return jnp.squeeze(critic1, -1), jnp.squeeze(critic2, -1)


class Actor(nn.Module):
    hidden_dims: typing.Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(
        self, observations: jnp.DeviceArray, temperature: float, rng: PRNGKey
    ) -> typing.Tuple[jnp.DeviceArray, jnp.DeviceArray, PRNGKey]:
        outputs = MLP((*self.hidden_dims, 2 * self.action_dim))(observations)

        means, log_stds = jnp.split(outputs, 2, axis=-1)
        log_stds = jnp.clip(log_stds, -5.0, 2.0)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        rng, key = jax.random.split(rng)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        return actions, log_probs, rng


def update_actor(
    actor_def: nn.Module, critic_def: nn.Module, actor_optimizer: Optimizer,
    alpha_optimizer: Optimizer, critic_params: Params, batch: rl_types.Batch,
    target_entropy: float, key: PRNGKey
) -> typing.Tuple[Optimizer, Optimizer, typing.Dict[str, float]]:
    alpha = jnp.exp(alpha_optimizer.target)

    def actor_loss_fn(actor_params):
        actions, log_probs, _ = actor_def.apply({'params': actor_params},
                                                batch.observations, 1.0, key)
        q1, q2 = critic_def.apply({'params': critic_params},
                                  batch.observations, actions)
        q = jnp.minimum(q1, q2)
        return (log_probs * alpha - q).mean(), log_probs

    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (actor_loss, log_probs), actor_grad = actor_grad_fn(actor_optimizer.target)
    actor_optimizer = actor_optimizer.apply_gradient(actor_grad)

    def alpha_loss_fn(log_alpha):
        return log_alpha * (-log_probs - target_entropy).mean()

    alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
    alpha_loss, alpha_grad = alpha_grad_fn(alpha_optimizer.target)
    alpha_optimizer = alpha_optimizer.apply_gradient(alpha_grad)

    return (actor_optimizer, alpha_optimizer, {
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'entropy': -log_probs.mean()
    })


def update_critic(
        actor_def: nn.Module, critic_def: nn.Module,
        critic_optimizer: Optimizer, actor_params: Params, alpha: float,
        target_critic_params: Params, batch: rl_types.Batch, tau: float,
        discount: float, key: PRNGKey
) -> typing.Tuple[Optimizer, Params, typing.Dict[str, float]]:
    next_actions, next_log_probs, _ = actor_def.apply({'params': actor_params},
                                                      batch.next_observations,
                                                      1.0, key)

    next_q1, next_q2 = critic_def.apply({'params': target_critic_params},
                                        batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.discounts * (
        next_q - alpha * next_log_probs)

    def critic_loss_fn(critic_params):
        q1, q2 = critic_def.apply({'params': critic_params},
                                  batch.observations, batch.actions)
        return ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

    critic_grad_fn = jax.value_and_grad(critic_loss_fn)
    critic_loss, critic_grad = critic_grad_fn(critic_optimizer.target)
    critic_optimizer = critic_optimizer.apply_gradient(critic_grad)

    def soft_update(params: Params, target_params: Params,
                    tau: float) -> Params:
        return jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau),
                                 params, target_params)

    target_critic_params = soft_update(critic_optimizer.target,
                                       target_critic_params, tau)

    return critic_optimizer, target_critic_params, {'critic_loss': critic_loss}


@jax.partial(jax.jit, static_argnums=(0, 1))
def update_step_jit(
    actor_def: nn.Module, critic_def: nn.Module, actor_optimizer: Optimizer,
    critic_optimizer: Optimizer, alpha_optimizer: Optimizer,
    target_critic_params: Params, batch: rl_types.Batch, tau: float,
    discount: float, target_entropy: float, rng: PRNGKey
) -> typing.Tuple[Optimizer, Optimizer, Optimizer, Params, typing.Dict[
        str, float], PRNGKey]:

    rng, key = jax.random.split(rng)
    actor_optimizer, alpha_optimizer, actor_info = update_actor(
        actor_def, critic_def, actor_optimizer, alpha_optimizer,
        critic_optimizer.target, batch, target_entropy, key)

    rng, key = jax.random.split(rng)
    alpha = jnp.exp(alpha_optimizer.target)
    critic_optimizer, target_critic_params, critic_info = update_critic(
        actor_def, critic_def, critic_optimizer, actor_optimizer.target, alpha,
        target_critic_params, batch, tau, discount, key)

    return (actor_optimizer, critic_optimizer, alpha_optimizer,
            target_critic_params, {
                **critic_info,
                **actor_info
            }, rng)


class SAC(object):
    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 seed: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 discount: float = 0.99,
                 tau: float = 0.005):

        self.discount = discount
        self.tau = tau

        self.rng = jax.random.PRNGKey(seed)

        self.actor_def = Actor((256, 256), action_dim)
        self.critic_def = DoubleCritic((256, 256))
        self.actor_apply_jit = jax.jit(self.actor_def.apply)

        observation_inputs = jnp.zeros((1, observation_dim))
        action_inputs = jnp.zeros((1, action_dim))

        self.rng, key = jax.random.split(self.rng)
        actor_params = self.actor_def.init(key, observation_inputs, 1.0,
                                           key)['params']
        self.rng, key = jax.random.split(self.rng)
        critic_params = self.critic_def.init(key, observation_inputs,
                                             action_inputs)['params']
        self.target_critic_params = copy.deepcopy(critic_params)
        log_alpha = jnp.log(0.1)

        self.critic_optimizer = flax.optim.Adam(
            learning_rate=critic_lr).create(critic_params)
        self.actor_optimizer = flax.optim.Adam(
            learning_rate=actor_lr).create(actor_params)
        self.alpha_optimizer = flax.optim.Adam(
            learning_rate=alpha_lr).create(log_alpha)

        self.target_entropy = -action_dim

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        actions, _, self.rng = self.actor_apply_jit(
            {'params': self.actor_optimizer.target}, observations, temperature,
            self.rng)
        return np.asarray(actions)

    def update_step(self, batch: rl_types.Batch) -> typing.Dict[str, float]:
        (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
         self.target_critic_params, info, self.rng) = update_step_jit(
             self.actor_def, self.critic_def, self.actor_optimizer,
             self.critic_optimizer, self.alpha_optimizer,
             self.target_critic_params, batch, self.tau, self.discount,
             self.target_entropy, self.rng)

        return info
