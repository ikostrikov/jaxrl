import typing

import gym
import numpy as np

TimeStep = typing.Tuple[np.ndarray, float, bool, dict]


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_sum = 0.0
        self.episode_length = 0

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1

        info['episode_return'] = self.reward_sum
        info['episode_length'] = self.episode_length

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self.reward_sum = 0
        self.episode_length = 0
        return self.env.reset()