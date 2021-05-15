import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.distribution = 'made'  # mog or made

    config.actor_lr = 1e-3
    config.hidden_dims = (256, 256)

    return config
