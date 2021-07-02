import copy

import gym


class TakeKey(gym.Wrapper):
    def __init__(self, env, take_key):
        super(TakeKey, self).__init__(env)
        self._take_key = take_key

        assert take_key in self.observation_space.spaces
        self.observation_space = self.env.observation_space[take_key]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation[self._take_key]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = copy.copy(observation)

        taken_observation = observation.pop(self._take_key)

        info['ignored_observations'] = observation

        return taken_observation, reward, done, info