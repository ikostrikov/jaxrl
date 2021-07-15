import gym
import numpy as np


class RGB2Gray(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(*obs_shape[:2], 1),
                                                dtype=np.uint8)

    def observation(self, observation):
        observation = np.dot(observation, [[0.299], [0.587], [0.114]])
        return observation.astype(np.uint8)