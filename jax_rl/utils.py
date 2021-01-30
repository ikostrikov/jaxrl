from typing import Optional

import dmc2gym
import gym
from gym.wrappers import RescaleAction

from jax_rl import wrappers
from jax_rl.wrappers import VideoRecorder


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None) -> gym.Env:
    if env_name.startswith('dmc'):
        domain_name, task_name = env_name.split('-')[1:]
        env = dmc2gym.make(domain_name=domain_name,
                           task_name=task_name,
                           seed=seed,
                           visualize_reward=True)
    else:
        env = gym.make(env_name)
        env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    env = wrappers.SinglePrecision(env)
    env = wrappers.EpisodeMonitor(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env