import ml_collections


def get_config():
    # https://arxiv.org/abs/2101.05982
    config = ml_collections.ConfigDict()

    config.algo = 'redq'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.n = 10
    config.m = 2

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.backup_entropy = True

    config.replay_buffer_size = None

    return config
