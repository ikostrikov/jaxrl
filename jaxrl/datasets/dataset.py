import collections

import numpy as np
from tqdm import tqdm

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 next_observations: np.ndarray, size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

    def take_top(self, percentile: float = 100.0):
        assert percentile > 0.0 and percentile <= 100.0

        trajs = [[]]

        for i in tqdm(range(len(self.observations))):
            if i > 0:
                # Detect breaks between trajectories.
                norm = np.linalg.norm(self.next_observations[i - 1] -
                                      self.observations[i])
                if norm > 0.0 or self.masks[i - 1] == 0.0:
                    trajs.append([])

            trajs[-1].append(
                (self.observations[i], self.actions[i], self.rewards[i],
                 self.masks[i], self.next_observations[i]))

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        N = int(len(trajs) * percentile / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        observations = []
        actions = []
        rewards = []
        masks = []
        next_observations = []

        for traj in trajs:
            for (obs, act, rew, mask, next_obs) in traj:
                observations.append(obs)
                actions.append(act)
                rewards.append(rew)
                masks.append(mask)
                next_observations.append(next_obs)

        self.observations = np.stack(observations)
        self.actions = np.stack(actions)
        self.rewards = np.stack(rewards)
        self.masks = np.stack(masks)
        self.next_observations = np.stack(next_observations)
        self.size = len(self.observations)
