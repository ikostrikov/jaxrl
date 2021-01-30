import flax

from jax_rl.networks.common import Model, PRNGKey


@flax.struct.dataclass
class ActorCriticTemp:
    actor: Model
    critic: Model
    target_critic: Model
    temp: Model
    rng: PRNGKey
