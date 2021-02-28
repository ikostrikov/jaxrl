"""Implementations of algorithms for continuous control."""

import copy
import typing
from functools import partial

import flax
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

import replay_buffer

tfd = tfp.distributions
tfb = tfp.bijectors

PRNGKey = typing.Any
Params = flax.core.frozen_dict.FrozenDict
InfoDict = typing.Dict[str, float]

default_init = hk.initializers.Orthogonal


class TrainingState(typing.NamedTuple):
    params: hk.Params
    opt_state: typing.Any


class State(typing.NamedTuple):
    actor: TrainingState
    critic: TrainingState
    target_critic: TrainingState
    log_alpha: TrainingState
    rng: PRNGKey
    step: int = 0


class DoubleCritic(hk.Module):
    def __init__(self, hidden_dims: typing.Sequence[int]):
        super().__init__()
        self.critic1 = hk.nets.MLP((*hidden_dims, 1), w_init=default_init())
        self.critic2 = hk.nets.MLP((*hidden_dims, 1), w_init=default_init())

    def __call__(
            self, observations: jnp.ndarray,
            actions: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
        inputs = jnp.concatenate([observations, actions], -1)
        critic1 = self.critic1(inputs)
        critic2 = self.critic2(inputs)
        return jnp.squeeze(critic1, -1), jnp.squeeze(critic2, -1)


class Actor(hk.Module):
    def __init__(self, hidden_dims: typing.Sequence[int], action_dim: int):
        super().__init__()

        self.dist_net = hk.nets.MLP(hidden_dims,
                                    activate_final=True,
                                    w_init=default_init())
        self.mean_linear = hk.Linear(action_dim, w_init=default_init())
        self.log_std_linear = hk.Linear(action_dim, w_init=default_init(1e-3))

    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.TransformedDistribution:
        features = self.dist_net(observations)
        means = self.mean_linear(features)
        log_stds = self.log_std_linear(features)

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
    """Stateless learner class for Soft-Actor-Critic."""
    def __init__(self, action_dim: int, config: ml_collections.ConfigDict):

        self.config = config
        self.actor_net = hk.without_apply_rng(
            hk.transform(lambda *args: Actor((256, 256), action_dim)(*args)))
        self.critic_net = hk.without_apply_rng(
            hk.transform(lambda *args: DoubleCritic((256, 256))(*args)))

        self.actor_opt = optax.adam(3e-4)
        self.critic_opt = optax.adam(3e-4)
        self.log_alpha_opt = optax.adam(3e-4)

    def initial_state(self, seed: int, observation_dim: int,
                      action_dim: int) -> State:
        rng = jax.random.PRNGKey(seed)

        observations = jnp.zeros((1, observation_dim))
        actions = jnp.zeros((1, action_dim))

        rng, actor_key = jax.random.split(rng)
        actor_params = self.actor_net.init(actor_key, observations)
        actor_opt_state = self.actor_opt.init(actor_params)
        actor_state = TrainingState(actor_params, actor_opt_state)

        rng, critic_key = jax.random.split(rng)
        critic_params = self.critic_net.init(critic_key, observations, actions)
        critic_opt_state = self.critic_opt.init(critic_params)
        critic_state = TrainingState(critic_params, critic_opt_state)

        target_critic_params = copy.deepcopy(critic_params)
        target_critic_state = TrainingState(target_critic_params, None)

        log_alpha = jnp.log(1.0)
        log_alpha_opt_state = self.log_alpha_opt.init(log_alpha)
        log_alpha_state = TrainingState(log_alpha, log_alpha_opt_state)

        return State(actor=actor_state,
                     critic=critic_state,
                     target_critic=target_critic_state,
                     log_alpha=log_alpha_state,
                     rng=rng)

    @jax.partial(jax.jit, static_argnums=0)
    def _sample_actions(self,
                        rng: PRNGKey,
                        actor_params: hk.Params,
                        observations: np.ndarray,
                        temperature: float = 1.0) -> jnp.ndarray:
        dist = self.actor_net.apply(actor_params, observations, temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)

    def sample_action(self,
                      state: State,
                      observation: np.ndarray,
                      temperature: float = 1.0) -> np.ndarray:
        rng, actions = self._sample_actions(state.rng, state.actor.params,
                                            observation[np.newaxis],
                                            temperature)
        actions = np.asarray(actions)
        action = np.clip(actions[0], -1, 1)
        state = state._replace(rng=rng)
        return state, action

    def _update_critic(
            self, state: State,
            batch: replay_buffer.Batch) -> typing.Tuple[State, InfoDict]:

        alpha = jnp.exp(state.log_alpha.params)

        rng, key = jax.random.split(state.rng)
        dist = self.actor_net.apply(state.actor.params,
                                    batch.next_observations)
        next_actions = dist.sample(seed=key)
        next_log_probs = dist.log_prob(next_actions)

        next_q1, next_q2 = self.critic_net.apply(state.target_critic.params,
                                                 batch.next_observations,
                                                 next_actions)
        next_q = jnp.minimum(next_q1, next_q2)

        target_q = batch.rewards + self.config.discount * batch.masks * (
            next_q - alpha * next_log_probs)

        def critic_loss_fn(critic_params):
            q1, q2 = self.critic_net.apply(critic_params, batch.observations,
                                           batch.actions)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            return critic_loss, (q1.mean(), q2.mean())

        critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        (critic_loss, (q1,
                       q2)), critic_grad = critic_grad_fn(state.critic.params)

        critic_update, critic_opt_state = self.critic_opt.update(
            critic_grad, state.critic.opt_state)
        critic_params = optax.apply_updates(state.critic.params, critic_update)

        target_critic_params = soft_update(critic_params,
                                           state.target_critic.params,
                                           self.config.tau)

        state = state._replace(critic=TrainingState(critic_params,
                                                    critic_opt_state),
                               target_critic=TrainingState(
                                   target_critic_params, None),
                               rng=rng)

        return state, {'critic_loss': critic_loss, 'q1': q1, 'q2': q2}

    def _update_actor(
            self, state: State,
            batch: replay_buffer.Batch) -> typing.Tuple[State, InfoDict]:
        alpha = jnp.exp(state.log_alpha.params)

        rng, key = jax.random.split(state.rng)

        def actor_loss_fn(actor_params):
            dist = self.actor_net.apply(actor_params, batch.observations)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)

            q1, q2 = self.critic_net.apply(state.critic.params,
                                           batch.observations, actions)
            q = jnp.minimum(q1, q2)
            actor_loss = (log_probs * alpha - q).mean()
            return actor_loss, -log_probs.mean()

        actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, entropy), actor_grad = actor_grad_fn(state.actor.params)

        actor_update, actor_opt_state = self.actor_opt.update(
            actor_grad, state.actor.opt_state)
        actor_params = optax.apply_updates(state.actor.params, actor_update)

        state = state._replace(actor=TrainingState(actor_params,
                                                   actor_opt_state),
                               rng=rng)

        return (state, {
            'actor_loss': actor_loss,
            'alpha': alpha,
            'entropy': entropy
        })

    def _update_alpha(self, state: State,
                      entropy: float) -> typing.Tuple[State, InfoDict]:
        def log_alpha_loss_fn(log_alpha):
            return jnp.exp(log_alpha) * (entropy -
                                         self.config.target_entropy).mean()

        log_alpha_grad_fn = jax.value_and_grad(log_alpha_loss_fn)
        log_alpha_loss, log_alpha_grad = log_alpha_grad_fn(
            state.log_alpha.params)

        log_alpha_update, log_alpha_opt_state = self.log_alpha_opt.update(
            log_alpha_grad, state.log_alpha.opt_state)
        log_alpha = optax.apply_updates(state.log_alpha.params,
                                        log_alpha_update)

        state = state._replace(
            log_alpha=TrainingState(log_alpha, log_alpha_opt_state))

        return state, {
            'log_alpha_loss': log_alpha_loss,
            'alpha': jnp.exp(log_alpha)
        }

    @jax.partial(jax.jit, static_argnums=0)
    def update(self, state: State,
               batch: replay_buffer.Batch) -> typing.Tuple[State, InfoDict]:

        state, critic_info = self._update_critic(state, batch)
        state, actor_info = self._update_actor(state, batch)
        state, alpha_info = self._update_alpha(state, actor_info['entropy'])

        state = state._replace(step=state.step + 1)

        return state, {**critic_info, **actor_info, **alpha_info}
