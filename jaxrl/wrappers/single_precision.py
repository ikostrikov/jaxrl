import gym
import numpy as np
from gym.spaces import Box


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.observation_space, Box)

        obs_space = self.observation_space
        self.observation_space = Box(obs_space.low, obs_space.high,
                                     obs_space.shape)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32)
