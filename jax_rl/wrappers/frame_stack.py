import collections

import gym
import numpy as np
from gym.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stack_axis=-1):
        super().__init__(env)
        self._num_stack = num_stack
        self._stack_axis = stack_axis

        self._frames = collections.deque([], maxlen=num_stack)

        low = np.repeat(self.observation_space.low, num_stack, axis=stack_axis)
        high = np.repeat(self.observation_space.high,
                         num_stack,
                         axis=stack_axis)
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._num_stack
        return np.concatenate(list(self._frames), axis=self._stack_axis)
