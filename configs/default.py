import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.max_steps = int(1e6)
    config.batch_size = 256
    config.start_training = int(1e4)

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.alpha_lr = 3e-4

    config.discount = 0.99
    config.tau = 0.005

    return config
