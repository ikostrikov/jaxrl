from configs import sac_default as default_lib


def get_config():
    config = default_lib.get_config()

    config.tau = 1.0
    config.target_update_period = 50

    return config
