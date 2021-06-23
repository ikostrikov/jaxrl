"""
Take from
https://github.com/openai/atari-reset/blob/master/atari_reset/wrappers.py
"""

import gym
import numpy as np


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info