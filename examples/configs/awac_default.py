import ml_collections


# Hyper parameters from the official implementation.
def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'awac'

    config.actor_optim_kwargs = ml_collections.ConfigDict()
    config.actor_optim_kwargs.learning_rate = 3e-4
    config.actor_optim_kwargs.weight_decay = 1e-4
    config.actor_hidden_dims = (256, 256, 256, 256)
    config.state_dependent_std = False

    config.critic_lr = 3e-4
    config.critic_hidden_dims = (256, 256)
    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.beta = 2.0

    config.num_samples = 1

    config.replay_buffer_size = None

    return config
