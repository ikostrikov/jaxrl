import copy
import os
import random

import numpy as np
import tqdm
from absl import app, flags
from gym import spaces
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import AWACLearner, SACLearner, SACV1Learner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'FetchPush-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_enum('goal_sampling', 'future', ['final', 'future'],
                  'Sampling strategy.')
flags.DEFINE_integer('num_goals', 1, 'Number of goals to sample.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
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
    if FLAGS.num_goals > 1:
        assert FLAGS.goal_sampling != 'final'

    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name,
                   FLAGS.seed,
                   video_train_folder,
                   flatten=False)
    eval_env = make_env(FLAGS.env_name,
                        FLAGS.seed + 42,
                        video_eval_folder,
                        flatten=True)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    flat_observation_space = spaces.flatten_space(env.observation_space)
    if algo == 'sac':
        agent = SACLearner(FLAGS.seed,
                           flat_observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'sac_v1':
        agent = SACV1Learner(FLAGS.seed,
                             flat_observation_space.sample()[np.newaxis],
                             env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(FLAGS.seed,
                            flat_observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(
        flat_observation_space, action_dim, replay_buffer_size
        or FLAGS.max_steps * (FLAGS.num_goals + 1))

    eval_returns = []
    observation_dict, done, trajectory = env.reset(), False, []
    observation = spaces.flatten(env.observation_space, observation_dict)
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation_dict, reward, done, info = env.step(action)
        next_observation = spaces.flatten(env.observation_space,
                                          next_observation_dict)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask,
                             next_observation)
        trajectory.append(
            copy.deepcopy(
                (observation_dict, action, mask, next_observation_dict, info)))
        observation = next_observation
        observation_dict = next_observation_dict

        if done:
            for _ in range(FLAGS.num_goals):
                for t_i, (obs, act, mas, next_obs,
                          info) in enumerate(trajectory):
                    if FLAGS.goal_sampling == 'final':
                        ind = -1
                    else:
                        ind = random.randint(t_i, len(trajectory) - 1)
                    final_goal = trajectory[ind][-2][
                        'achieved_goal']  # Next observation achieved goal.
                    rew = env.compute_reward(next_obs['achieved_goal'],
                                             final_goal, info)
                    obs['desired_goal'] = final_goal
                    next_obs['desired_goal'] = final_goal
                    obs = spaces.flatten(env.observation_space, obs)
                    next_obs = spaces.flatten(env.observation_space, next_obs)
                    replay_buffer.insert(obs, act, rew, mas, next_obs)

            observation_dict, done, trajectory = env.reset(), False, []
            observation = spaces.flatten(env.observation_space,
                                         observation_dict)
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

            summary_writer.add_scalar(f'training/success', info['is_success'],
                                      info['total']['timesteps'])

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent,
                                  eval_env,
                                  FLAGS.eval_episodes,
                                  with_success=True)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
