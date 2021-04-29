import gym


class TakeKey(gym.ObservationWrapper):
    def __init__(self, env, take_key):
        super(TakeKey, self).__init__(env)
        self._take_key = take_key
        self.observation_space = self.env.observation_space[take_key]

    def observation(self, observation):
        return observation[self._take_key]