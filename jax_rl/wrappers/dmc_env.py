# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

from typing import Dict, Optional

import numpy as np
from dm_control import suite
from dm_env import specs
from gym import core, spaces

from jax_rl.wrappers.common import TimeStep


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCEnv(core.Env):
    def __init__(self,
                 domain_name: str,
                 task_name: str,
                 task_kwargs: Optional[Dict] = {},
                 environment_kwargs=None):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'

        self._env = suite.load(domain_name=domain_name,
                               task_name=task_name,
                               task_kwargs=task_kwargs,
                               environment_kwargs=environment_kwargs)

        self.action_space = _spec_to_box([self._env.action_spec()])

        self.observation_space = _spec_to_box(
            self._env.observation_spec().values())

        self.seed(seed=task_kwargs['random'])

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = _flatten_obs(time_step.observation)

        info = {}
        if done and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True

        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return _flatten_obs(time_step.observation)

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(height=height,
                                        width=width,
                                        camera_id=camera_id)
