__all__ = ['seed_all', 'seed_env']

from typing import Optional
import gym


def seed_all(envs: Optional[gym.Env] = None, seed: Optional[int] = None):
    import random
    import numpy as np
    import torch as th
    """Seed everything!"""
    if seed is None:
        import sys # Generate random seed
        seed = random.randrange(sys.maxsize)
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    if envs is not None: seed_env(envs, seed)
    return seed


def seed_env(env: gym.Env, seed: int):
    env.seed(seed)
    env.action_space.seed()
    env.observation_space.seed()