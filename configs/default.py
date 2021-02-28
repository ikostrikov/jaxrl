import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.alpha_lr = 3e-4

    config.discount = 0.99
    config.tau = 0.005

    return config
