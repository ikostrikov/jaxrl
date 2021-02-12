import numpy as np

import rl_types


class ReplayBuffer(object):
    def __init__(self, observation_dim: int, action_dim: int, capacity: int):
        self.observations = np.empty((capacity, observation_dim),
                                     dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, ), dtype=np.float32)
        self.masks = np.empty((capacity, ), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim),
                                          dtype=np.float32)

        self.insert_index = 0
        self.size = 0

        self.capacity = capacity

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, discount: float, next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = discount
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> rl_types.Batch:
        ind = np.random.randint(low=0, high=self.size, size=(batch_size, ))
        return rl_types.Batch(observations=self.observations[ind],
                              actions=self.actions[ind],
                              rewards=self.rewards[ind],
                              masks=self.masks[ind],
                              next_observations=self.next_observations[ind])

    def __len__(self):
        return self.size
