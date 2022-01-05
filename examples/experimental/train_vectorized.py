import os
import random

import numpy as np
import tqdm
from absl import app, flags
from gym.vector.async_vector_env import AsyncVectorEnv
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import jaxrl.utils as env_utils
from jaxrl.agents import SACLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('num_envs', 8, 'Number of parallel environments.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    if FLAGS.save_video:
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_eval_folder = None

    def make_env(env_name, seed):

        def _make():
            env = env_utils.make_env(env_name, seed, None)
            return env

        return _make

    env_fns = [
        make_env(FLAGS.env_name, FLAGS.seed + i) for i in range(FLAGS.num_envs)
    ]

    env = AsyncVectorEnv(env_fns=env_fns)
    eval_env = env_utils.make_env(FLAGS.env_name, FLAGS.seed + 42,
                                  video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    agent = SACLearner(FLAGS.seed,
                       eval_env.observation_space.sample()[np.newaxis],
                       eval_env.action_space.sample()[np.newaxis], **kwargs)

    replay_buffer = ReplayBuffer(eval_env.observation_space,
                                 eval_env.action_space, replay_buffer_size
                                 or FLAGS.max_steps)

    eval_returns = []
    observation = env.reset()
    dones = [False] * FLAGS.num_envs
    infos = [{}] * FLAGS.num_envs
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i % FLAGS.num_envs == 0:
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action = agent.sample_actions(observation)
            next_observation, reward, dones, infos = env.step(action)

            # TODO: Add batch insert.
            for j in range(FLAGS.num_envs):
                replay_buffer.insert(observation[j], action[j],
                                     reward[j], 1.0 - float(dones[j]),
                                     float(dones[j]), next_observation[j])
            observation = next_observation

            total_timesteps = 0
            for info in infos:
                total_timesteps += info['total']['timesteps']

            for j, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    for k, v in info['episode'].items():
                        summary_writer.add_scalar(f'training/{k}_{j}', v,
                                                  total_timesteps)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          total_timesteps)
            summary_writer.flush()

            eval_returns.append((total_timesteps, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
