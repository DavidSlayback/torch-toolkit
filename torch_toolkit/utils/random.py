__all__ = ['seed_all', 'seed_env']

from typing import Optional
import gym


def seed_all(envs: Optional[gym.Env] = None, seed: Optional[int] = None):
    """Set all necessary random seeds

    Seeds torch, random, and numpy with the same seed

    Args:
        envs: If provided, also seed the given gym Env
        seed: If provided, use this as the seed value. Otherwise, autogenerate one
    """
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
    """Set env random seed, action space seed, obs space seed

    Args:
        env: gym.Env instance to seed
        seed: Integer seed value, which should be chosen or generated elsewhere
    """
    env.seed(seed)
    env.action_space.seed()
    env.observation_space.seed()