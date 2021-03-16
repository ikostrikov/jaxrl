from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from jax_rl.networks.common import MLP, Params, PRNGKey, default_init

tfd = tfp.distributions
tfb = tfp.bijectors


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.TransformedDistribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(1e-3))(outputs)
        log_stds = nn.Dense(self.action_dim,
                            kernel_init=default_init(1e-3))(outputs)

        log_stds = jnp.clip(log_stds, -20.0, 2.0)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        return tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.TransformedDistribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init(1e-3))(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(1e-3),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init(1e-3))(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)
        log_stds = jnp.clip(log_stds, -20.0, 2.0)

        components_distribution = tfd.TransformedDistribution(
            tfd.Normal(loc=mu, scale=jnp.exp(log_stds) * temperature),
            tfp.bijectors.Tanh())

        distribution = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=components_distribution)

        return tfd.Independent(distribution)


@jax.partial(jax.jit, static_argnums=1)
def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
