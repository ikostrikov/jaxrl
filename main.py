import os
import random

import gym
import haiku as hk
import jax
import numpy as np
import tqdm
from absl import app, flags
from gym.wrappers.rescale_action import RescaleAction
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from replay_buffer import ReplayBuffer
from sac import SAC

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', '/tmp/sac/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'configs/default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(_):
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    def wrap(env):
        env = wrappers.SinglePrecision(env)
        env = RescaleAction(env, -1.0, 1.0)
        return wrappers.EpisodeMonitor(env)

    env = wrap(gym.make(FLAGS.env_name))
    eval_env = wrap(gym.make(FLAGS.env_name))

    for e in [eval_env, env]:
        e.seed(FLAGS.seed)
        e.action_space.seed(FLAGS.seed)
        e.observation_space.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    FLAGS.config.unlock()
    FLAGS.config.target_entropy = -action_dim / 2

    agent = SAC(action_dim, FLAGS.config)
    state = agent.initial_state(FLAGS.seed, observation_dim, action_dim)

    replay_buffer = ReplayBuffer(observation_dim, action_dim, FLAGS.max_steps)

    eval_returns = []
    done, info, observation = True, {}, np.empty(())
    for i in tqdm.tqdm(range(FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if done:
            observation = env.reset()
            done = False
            for k in ['episode_return', 'episode_length', 'episode_duration']:
                if k in info:
                    summary_writer.add_scalar(f'training/{k}', info[k], i)

        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            state, action = agent.sample_action(state, observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                             next_observation)
        observation = next_observation

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            state, update_info = agent.update(state, batch)

            if (i + 1) % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)

            summary_writer.flush()

        if (i + 1) % FLAGS.eval_interval == 0:
            eval_stats = {'episode_return': [], 'episode_length': []}
            for _ in range(FLAGS.eval_episodes):
                eval_observation = eval_env.reset()
                while True:
                    state, action = agent.sample_action(state,
                                                        eval_observation,
                                                        temperature=0.0)
                    eval_observation, _, eval_done, eval_info = eval_env.step(
                        action)
                    if eval_done:
                        for k in eval_stats.keys():
                            eval_stats[k].append(eval_info[k])
                        break
            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s',
                                          np.mean(v), i)
            eval_returns.append((i + 1, np.mean(eval_stats['episode_return'])))

            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns)


if __name__ == '__main__':
    app.run(main)
