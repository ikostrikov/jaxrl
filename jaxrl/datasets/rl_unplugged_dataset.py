import os

import d4rl
import numpy as np

from jaxrl.datasets.dataset import Dataset


class RLUnpluggedDataset(Dataset):

    def __init__(self,
                 task_name: str,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        save_dir = os.path.join(d4rl.offline_env.DATASET_PATH, 'rl_unplugged')
        os.makedirs(save_dir, exist_ok=True)
        dataset = {}
        with open(os.path.join(save_dir, f'{task_name}.npz'), 'rb') as f:
            dataset_file = np.load(f)
            for k, v in dataset_file.items():
                dataset[k] = v
        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=dataset['masks'].astype(np.float32),
                         dones_float=dataset['done_floats'].astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))
