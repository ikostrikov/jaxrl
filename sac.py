"""Implementations of algorithms for continuous control."""

import copy
import typing
from functools import partial

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


class State(typing.NamedTuple):
    actor_optimizer: Optimizer
    critic_optimizer: Optimizer
    target_critic_params: Params
    log_alpha_optimizer: Optimizer
    rng: PRNGKey
    step: int = 0


default_init = nn.initializers.orthogonal


class MLP(nn.Module):
    hidden_dims: typing.Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for size in self.hidden_dims[:-1]:
            x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.relu(x)
        x = nn.Dense(self.hidden_dims[-1], kernel_init=default_init())(x)

        return x


class DoubleCritic(nn.Module):
    hidden_dims: typing.Sequence[int]

    @nn.compact
    def __call__(
            self, observations: jnp.ndarray,
            actions: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
        inputs = jnp.concatenate([observations, actions], -1)
        critic1 = MLP((*self.hidden_dims, 1))(inputs)
        critic2 = MLP((*self.hidden_dims, 1))(inputs)
        return jnp.squeeze(critic1, -1), jnp.squeeze(critic2, -1)


class Actor(nn.Module):
    hidden_dims: typing.Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.TransformedDistribution:
        outputs = nn.relu(MLP(self.hidden_dims)(observations))
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = nn.Dense(self.action_dim,
                            kernel_init=default_init(1e-3))(outputs)

        log_stds = jnp.clip(log_stds, -20.0, 2.0)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        return tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())


def soft_update(params: Params, target_params: Params, tau: float) -> Params:
    return jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), params,
                             target_params)


class SAC(object):
    def __init__(self, action_dim: int, config: ml_collections.ConfigDict):
        self.actor_def = Actor((256, 256), action_dim)
        self.critic_def = DoubleCritic((256, 256))

        config = copy.deepcopy(config).unlock()
        config.target_entropy = -action_dim / 2
        self.config = ml_collections.FrozenConfigDict(config)

    def initial_state(self, seed: int, observation_dim: int,
                      action_dim: int) -> State:
        observation_inputs = jnp.zeros((1, observation_dim))
        action_inputs = jnp.zeros((1, action_dim))

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_params = self.actor_def.init(actor_key,
                                           observation_inputs)['params']
        critic_params = self.critic_def.init(critic_key, observation_inputs,
                                             action_inputs)['params']
        target_critic_params = copy.deepcopy(critic_params)
        log_alpha = jnp.log(1.0)

        return State(
            actor_optimizer=flax.optim.Adam(
                learning_rate=self.config.actor_lr).create(actor_params),
            critic_optimizer=flax.optim.Adam(
                learning_rate=self.config.critic_lr).create(critic_params),
            target_critic_params=target_critic_params,
            log_alpha_optimizer=flax.optim.Adam(
                learning_rate=self.config.alpha_lr).create(log_alpha),
            rng=rng)

    @jax.partial(jax.jit, static_argnums=0)
    def _sample_actions(self,
                        rng: PRNGKey,
                        actor_params: Params,
                        observations: np.ndarray,
                        temperature: float = 1.0) -> jnp.ndarray:
        dist = self.actor_def.apply({'params': actor_params}, observations,
                                    temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)

    def sample_action(self,
                      state: State,
                      observation: np.ndarray,
                      temperature: float = 1.0) -> np.ndarray:
        rng, actions = self._sample_actions(state.rng,
                                            state.actor_optimizer.target,
                                            observation[np.newaxis],
                                            temperature)
        actions = np.asarray(actions)
        action = np.clip(actions[0], -1, 1)
        state = state._replace(rng=rng)
        return state, action

    def _update_critic(
            self, state: State,
            batch: replay_buffer.Batch) -> typing.Tuple[State, InfoDict]:
        log_alpha = state.log_alpha_optimizer.target
        actor_params = state.actor_optimizer.target
        critic_params = state.critic_optimizer.target

        rng, key = jax.random.split(state.rng)
        dist = self.actor_def.apply({'params': actor_params},
                                    batch.next_observations, 1.0)
        next_actions = dist.sample(seed=key)
        next_log_probs = dist.log_prob(next_actions)

        next_q1, next_q2 = self.critic_def.apply(
            {'params': state.target_critic_params}, batch.next_observations,
            next_actions)
        next_q = jnp.minimum(next_q1, next_q2)

        target_q = batch.rewards + self.config.discount * batch.masks * (
            next_q - jnp.exp(log_alpha) * next_log_probs)

        def critic_loss_fn(critic_params):
            q1, q2 = self.critic_def.apply({'params': critic_params},
                                           batch.observations, batch.actions)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            return critic_loss, (q1.mean(), q2.mean())

        critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (critic_loss, (q1, q2)), critic_grad = critic_grad_fn(critic_params)
        critic_optimizer = state.critic_optimizer.apply_gradient(critic_grad)

        target_critic_params = soft_update(critic_optimizer.target,
                                           state.target_critic_params,
                                           self.config.tau)

        state = state._replace(critic_optimizer=critic_optimizer,
                               target_critic_params=target_critic_params,
                               rng=rng)

        return state, {'critic_loss': critic_loss, 'q1': q1, 'q2': q2}

    def _update_actor(
            self, state: State,
            batch: replay_buffer.Batch) -> typing.Tuple[State, InfoDict]:
        log_alpha = state.log_alpha_optimizer.target
        critic_params = state.critic_optimizer.target
        actor_params = state.actor_optimizer.target

        rng, key = jax.random.split(state.rng)

        def actor_loss_fn(actor_params):
            dist = self.actor_def.apply({'params': actor_params},
                                        batch.observations, 1.0)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            q1, q2 = self.critic_def.apply({'params': critic_params},
                                           batch.observations, actions)
            q = jnp.minimum(q1, q2)
            actor_loss = (log_probs * jnp.exp(log_alpha) - q).mean()
            return actor_loss, -log_probs.mean()

        actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, entropy), actor_grad = actor_grad_fn(actor_params)
        actor_optimizer = state.actor_optimizer.apply_gradient(actor_grad)

        state = state._replace(actor_optimizer=actor_optimizer, rng=rng)

        return (state, {'actor_loss': actor_loss, 'entropy': entropy})

    def _update_alpha(self, state: State,
                      entropy: float) -> typing.Tuple[State, InfoDict]:
        log_alpha = state.log_alpha_optimizer.target

        def alpha_loss_fn(log_alpha):
            return jnp.exp(log_alpha) * (entropy -
                                         self.config.target_entropy).mean()

        alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
        alpha_loss, alpha_grad = alpha_grad_fn(log_alpha)
        log_alpha_optimizer = state.log_alpha_optimizer.apply_gradient(
            alpha_grad)

        state = state._replace(log_alpha_optimizer=log_alpha_optimizer)

        return (state, {'alpha_loss': alpha_loss, 'alpha': jnp.exp(log_alpha)})

    @jax.partial(jax.jit, static_argnums=0)
    def update(self, state: State,
               batch: replay_buffer.Batch) -> typing.Tuple[State, InfoDict]:

        state, critic_info = self._update_critic(state, batch)
        state, actor_info = self._update_actor(state, batch)
        state, alpha_info = self._update_alpha(state, actor_info['entropy'])

        state = state._replace(step=state.step + 1)

        return state, {**critic_info, **actor_info, **alpha_info}
