import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'ddpg'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.replay_buffer_size = None

    config.exploration_noise = 0.1

    return config
