import copy

import gym


class TakeKey(gym.ObservationWrapper):

    def __init__(self, env, take_key):
        super(TakeKey, self).__init__(env)
        self._take_key = take_key

        assert take_key in self.observation_space.spaces
        self.observation_space = self.env.observation_space[take_key]

    def observation(self, observation):
        observation = copy.copy(observation)
        taken_observation = observation.pop(self._take_key)
        self._ignored_observations = observation
        return taken_observation
