import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'drq'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (32, 32, 32, 32)
    config.cnn_strides = (2, 1, 1, 1)
    config.cnn_padding = 'VALID'
    config.latent_dim = 50

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 0.1
    config.target_entropy = None

    config.replay_buffer_size = 100_000

    return config
