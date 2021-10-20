import os

import d4rl
import dm_control_suite
import gym
import numpy as np
from absl import app, flags
from scipy.spatial import cKDTree
from tqdm import tqdm

from jaxrl import wrappers

flags.DEFINE_string('path', '/home/kostrikov/datasets/', 'Path to dataset.')
flags.DEFINE_string('task_name', 'cheetah_run', 'Game.')
flags.DEFINE_enum('task_class', 'control_suite',
                  ['humanoid', 'rodent', 'control_suite'], 'Task classes.')

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.task_class == 'control_suite':
        task = dm_control_suite.ControlSuite(task_name=FLAGS.task_name)
    elif FLAGS.task_class == 'humanoid':
        task = dm_control_suite.CmuThirdParty(task_name=FLAGS.task_name)
    elif FLAGS.task_class == 'rodent':
        task = dm_control_suite.Rodent(task_name=FLAGS.task_name)

    environment = task.environment
    env = wrappers.DMCEnv(env=environment, task_kwargs={'random': 0})

    ds = dm_control_suite.dataset(root_path=FLAGS.path,
                                  data_path=task.data_path,
                                  shapes=task.shapes,
                                  num_threads=1,
                                  uint8_features=task.uint8_features,
                                  num_shards=100)
    observations = []
    actions = []
    rewards = []
    masks = []
    next_observations = []

    print("Reading the dataset")
    for i, sample in tqdm(enumerate(ds)):
        obs = gym.spaces.flatten(env.observation_space,
                                 sample.data[0]).astype(np.float32)
        action = gym.spaces.flatten(env.action_space,
                                    sample.data[1]).astype(np.float32)
        reward = float(sample.data[2].numpy().item())
        mask = float(sample.data[3].numpy().item())
        next_obs = gym.spaces.flatten(env.observation_space,
                                      sample.data[4]).astype(np.float32)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        masks.append(mask)
        next_observations.append(next_obs)

    # The datasets are shuffles even in the original files.
    # The code below unshuffles them.
    kdtree = cKDTree(observations)
    _, inds = kdtree.query(next_observations, distance_upper_bound=1e-5)

    kdtree = cKDTree(next_observations)
    dists_, _ = kdtree.query(observations, distance_upper_bound=1e-5)

    ordered_observations = []
    ordered_actions = []
    ordered_rewards = []
    ordered_masks = []
    ordered_next_observations = []

    print("Reordering the dataset")
    for i in tqdm(range(len(observations))):
        if dists_[i] > 0:
            j = i
            while j < len(observations):
                ordered_observations.append(observations[j])
                ordered_actions.append(actions[j])
                ordered_rewards.append(rewards[j])
                ordered_masks.append(masks[j])
                ordered_next_observations.append(next_observations[j])
                j = inds[j]

    print("Verifying the dataset")
    prev_i = -1

    ordered_done_floats = []
    for i in tqdm(range(len(ordered_observations))):
        if (i == len(ordered_observations) - 1
                or np.linalg.norm(ordered_observations[i + 1] -
                                  ordered_next_observations[i]) > 1e-5):
            assert i - prev_i == 1000
            prev_i = i
            ordered_done_floats.append(1.0)
        else:
            ordered_done_floats.append(0.0)

    print(f"Dataset size: {len(ordered_observations)}")

    ordered_observations = np.stack(ordered_observations)
    ordered_actions = np.stack(ordered_actions)
    ordered_rewards = np.stack(ordered_rewards)
    ordered_masks = np.stack(ordered_masks)
    ordered_done_floats = np.stack(ordered_done_floats)
    ordered_next_observations = np.stack(ordered_next_observations)

    save_dir = os.path.join(d4rl.offline_env.DATASET_PATH, 'rl_unplugged')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'{FLAGS.task_name}.npz'), 'wb') as f:
        np.savez_compressed(f,
                            observations=ordered_observations,
                            actions=ordered_actions,
                            rewards=ordered_rewards,
                            masks=ordered_masks,
                            done_floats=ordered_done_floats,
                            next_observations=ordered_next_observations)


if __name__ == '__main__':
    app.run(main)
