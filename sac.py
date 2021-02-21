"""Implementations of algorithms for continuous control."""

import copy
import typing

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from flax import linen as nn
from flax.optim.base import Optimizer
from tensorflow_probability.substrates import jax as tfp

import replay_buffer

tfd = tfp.distributions
tfb = tfp.bijectors

PRNGKey = typing.Any
Params = flax.core.frozen_dict.FrozenDict
InfoDict = typing.Dict[str, float]

default_init = nn.initializers.orthogonal


class State(typing.NamedTuple):
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer
    target_critic_params: Params
    alpha_optimizer: Optimizer
    rng: PRNGKey
    step: int = 0


class MLP(nn.Module):
    hidden_dims: typing.Sequence[int]
    final_dense_gain: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        for size in self.hidden_dims[:-1]:
            x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.relu(x)
        x = nn.Dense(self.hidden_dims[-1],
                     kernel_init=default_init(self.final_dense_gain))(x)

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
        outputs = MLP((*self.hidden_dims, 2 * self.action_dim),
                      final_dense_gain=1e-2)(observations)

        means, log_stds = jnp.split(outputs, 2, axis=-1)
        log_stds = jnp.clip(log_stds, -20.0, 2.0)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        rng, key = jax.random.split(rng)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        return actions, log_probs, rng


def update_actor(config: ml_collections.FrozenConfigDict,
                 actor_optimizer: Optimizer, alpha_optimizer: Optimizer,
                 critic_params: Params, batch: replay_buffer.Batch,
                 key: PRNGKey) -> typing.Tuple[Optimizer, Optimizer, InfoDict]:
    alpha = jnp.exp(alpha_optimizer.target)

    def actor_loss_fn(actor_params):
        actions, log_probs, _ = config.actor_def.apply(
            {'params': actor_params}, batch.observations, 1.0, key)
        q1, q2 = config.critic_def.apply({'params': critic_params},
                                         batch.observations, actions)
        q = jnp.minimum(q1, q2)
        return (log_probs * alpha - q).mean(), (-log_probs.mean(), q1.mean(),
                                                q2.mean())

    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (actor_loss, (entropy, q1,
                  q2)), actor_grad = actor_grad_fn(actor_optimizer.target)
    actor_optimizer = actor_optimizer.apply_gradient(actor_grad)

    def alpha_loss_fn(log_alpha):
        return jnp.exp(log_alpha) * (entropy - config.target_entropy).mean()

    alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
    alpha_loss, alpha_grad = alpha_grad_fn(alpha_optimizer.target)
    alpha_optimizer = alpha_optimizer.apply_gradient(alpha_grad)

    return (actor_optimizer, alpha_optimizer, {
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'alpha': alpha,
        'entropy': entropy,
        'q1': q1,
        'q2': q2
    })


def update_critic(config: ml_collections.FrozenConfigDict,
                  critic_optimizer: Optimizer, actor_params: Params,
                  alpha: float, target_critic_params: Params,
                  batch: replay_buffer.Batch,
                  key: PRNGKey) -> typing.Tuple[Optimizer, Params, InfoDict]:
    next_actions, next_log_probs, _ = config.actor_def.apply(
        {'params': actor_params}, batch.next_observations, 1.0, key)

    next_q1, next_q2 = config.critic_def.apply(
        {'params': target_critic_params}, batch.next_observations,
        next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + config.discount * batch.masks * (
        next_q - alpha * next_log_probs)

    def critic_loss_fn(critic_params):
        q1, q2 = config.critic_def.apply({'params': critic_params},
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
                                       target_critic_params, config.tau)

    return critic_optimizer, target_critic_params, {'critic_loss': critic_loss}


@jax.partial(jax.jit, static_argnums=0)
def update_step_jit(
        config: ml_collections.FrozenConfigDict, state: State,
        batch: replay_buffer.Batch) -> typing.Tuple[State, InfoDict]:

    rng, critic_key, actor_key = jax.random.split(state.rng, 3)

    alpha = jnp.exp(state.alpha_optimizer.target)
    critic_optimizer, target_critic_params, critic_info = update_critic(
        config, state.critic_optimizer, state.actor_optimizer.target, alpha,
        state.target_critic_params, batch, critic_key)

    actor_optimizer, alpha_optimizer, actor_info = update_actor(
        config, state.actor_optimizer, state.alpha_optimizer,
        critic_optimizer.target, batch, actor_key)

    return (State(actor_optimizer=actor_optimizer,
                  critic_optimizer=critic_optimizer,
                  target_critic_params=target_critic_params,
                  alpha_optimizer=alpha_optimizer,
                  rng=rng,
                  step=state.step + 1), {
                      **critic_info,
                      **actor_info
                  })


class SAC(object):
    def __init__(self, observation_dim: int, action_dim: int, seed: int,
                 config: ml_collections.ConfigDict):

        config = copy.deepcopy(config).unlock()
        config.actor_def = Actor((256, 256), action_dim)
        config.critic_def = DoubleCritic((256, 256))
        config.target_entropy = -action_dim / 2

        self.config = ml_collections.FrozenConfigDict(config)
        self.actor_apply_jit = jax.jit(self.config.actor_def.apply)

        observation_inputs = jnp.zeros((1, observation_dim))
        action_inputs = jnp.zeros((1, action_dim))

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, input_key = jax.random.split(rng, 4)
        actor_params = self.config.actor_def.init(actor_key,
                                                  observation_inputs, 1.0,
                                                  input_key)['params']
        critic_params = self.config.critic_def.init(critic_key,
                                                    observation_inputs,
                                                    action_inputs)['params']
        target_critic_params = copy.deepcopy(critic_params)
        log_alpha = jnp.log(1.0)

        self.rng, key = jax.random.split(rng)
        self.state = State(
            actor_optimizer=flax.optim.Adam(
                learning_rate=self.config.actor_lr).create(actor_params),
            critic_optimizer=flax.optim.Adam(
                learning_rate=self.config.critic_lr).create(critic_params),
            target_critic_params=target_critic_params,
            alpha_optimizer=flax.optim.Adam(
                learning_rate=self.config.alpha_lr).create(log_alpha),
            rng=key)

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        actions, _, self.rng = self.actor_apply_jit(
            {'params': self.state.actor_optimizer.target}, observations,
            temperature, self.rng)
        action = np.asarray(actions)
        return np.clip(action, -1.0, 1.0)

    def update_step(self, batch: replay_buffer.Batch) -> InfoDict:
        self.state, info = update_step_jit(self.config, self.state, batch)

        return info
