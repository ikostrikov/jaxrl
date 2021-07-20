from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module,
             env: gym.Env,
             num_episodes: int,
             with_success: bool = False) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    if with_success:
        successes = 0.0
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if with_success:
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if with_success:
        stats['success'] = successes / num_episodes
    return stats
