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


@jax.partial(jax.jit, static_argnums=1)
def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
