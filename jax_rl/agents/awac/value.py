from typing import Tuple

import jax
import jax.numpy as jnp

from jax_rl.agents.actor_critic_temp import ActorCriticTemp
from jax_rl.datasets import Batch


def get_value(models: ActorCriticTemp, batch: Batch, num_samples: int,
              soft_value: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = models.actor(batch.observations)

    rng, pi_key = jax.random.split(models.rng)
    policy_actions = dist.sample(seed=pi_key, sample_shape=[num_samples])
    log_probs = dist.log_prob(policy_actions)

    n_observations = jnp.repeat(batch.observations[jnp.newaxis],
                                num_samples,
                                axis=0)
    q_pi1, q_pi2 = models.critic(n_observations, policy_actions)

    def get_v(q):
        if soft_value:
            q = q / models.temp() - log_probs - jnp.log(q.shape[0])
            return models.temp() * jax.nn.logsumexp(q, axis=0)
        else:
            return jnp.mean(q, axis=0)

    new_models = models.replace(rng=rng)

    return new_models, get_v(q_pi1), get_v(q_pi2)
