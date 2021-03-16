import d4rl
import gym
import numpy as np

from jax_rl.datasets.dataset import Batch, Dataset


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        super().__init__(dataset['observations'],
                         actions=dataset['actions'],
                         rewards=dataset['rewards'],
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         next_observations=dataset['next_observations'],
                         size=len(dataset['observations']))
