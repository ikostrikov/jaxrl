from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import Model, PRNGKey


def get_value(key: PRNGKey, actor: Model, critic: Model, batch: Batch,
              num_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = actor(batch.observations)

    policy_actions = dist.sample(seed=key, sample_shape=[num_samples])

    n_observations = jnp.repeat(batch.observations[jnp.newaxis],
                                num_samples,
                                axis=0)
    q_pi1, q_pi2 = critic(n_observations, policy_actions)

    def get_v(q):
        return jnp.mean(q, axis=0)

    return get_v(q_pi1), get_v(q_pi2)
