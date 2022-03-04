__all__ = ['DictObservabilityWrapper', 'VectorObservabilityWrapper', 'DictVectorObservabilityWrapper']

from typing import Optional, Sequence, Union, Dict

import gym
import numpy as np


class DictObservabilityWrapper(gym.ObservationWrapper):
    """Strip componnents of dict observations that aren't in the list of fields

    Args:
        env: As in all wrappers
        fields: Sequence of dictionary keys to limit observations to. If None, use all fields
    """
    def __init__(self, env: gym.Env, fields: Optional[Sequence[Union[str, int]]] = None):
        super().__init__(env)
        os = getattr(self, 'single_observation_space', self.observation_space)
        assert isinstance(os, gym.spaces.Dict)
        self._obs_fields = fields or list(os.keys())
        new_os = gym.spaces.Dict({
                k: v for k, v in os.items() if k in self._obs_fields
        })
        if getattr(env, "is_vector_env", False):
            from gym.vector.utils import batch_space
            self.single_observation_space = new_os
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.observation_space = new_os


    def observation(self, observation):
        """Strip out some fields"""
        return {k: v for k, v in observation.items() if k in self._obs_fields}


class VectorObservabilityWrapper(gym.ObservationWrapper):
    """Strip components of vector observation that aren't in a sequence of indices

    Args:
        env:
        indexes: Sequence of indexes to limit observations to. If None, use all indices
    """
    def __init__(self, env: gym.Env, indexes: Optional[Sequence[int]] = None):
        super().__init__(env)
        os = getattr(self, 'single_observation_space', self.observation_space)
        assert isinstance(os, (gym.spaces.Box, gym.spaces.MultiBinary, gym.spaces.MultiDiscrete))
        assert len(os.shape) == 1  # Vectors only
        self._obs_indices = np.array(indexes) if (indexes is not None) else np.arange(os.shape[0])
        new_os = filter_gym_space(os, self._obs_indices)
        if getattr(env, "is_vector_env", False):
            from gym.vector.utils import batch_space
            self.single_observation_space = new_os
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else: self.observation_space = new_os


    def observation(self, observation):
        """Strip out some indices"""
        return observation[..., self._obs_indices]


def filter_gym_space(os: gym.Space, indexes: np.ndarray) -> gym.Space:
    """Filter out indexes of a particular gym space"""
    if isinstance(os, gym.spaces.Discrete): return os
    assert indexes.max() < os.shape[0]
    if isinstance(os, gym.spaces.Box):
        l, h = os.low[indexes], os.high[indexes]
        return gym.spaces.Box(l, h, indexes.shape, os.dtype)
    if isinstance(os, gym.spaces.MultiBinary): return gym.spaces.MultiBinary(indexes.shape[0])
    if isinstance(os, gym.spaces.MultiDiscrete): return gym.spaces.MultiDiscrete(os.nvec[indexes], os.dtype)
    raise ValueError("Unsupported space")


class DictVectorObservabilityWrapper(gym.ObservationWrapper):
    """Strip components of dict observations that a

    Args:
        env:
        kidx_pairs: Dict of {key: indexes} pairs of what should be observed. If None, defaults to all
    """
    def __init__(self, env: gym.Env, kidx_pairs: Dict[Union[str, int], Optional[Sequence[int]]] = None):
        super().__init__(env)
        os = getattr(self, 'single_observation_space', self.observation_space)
        assert isinstance(os, gym.spaces.Dict)
        # Strip down to proper dict components
        self._keys = list(kidx_pairs.keys())
        self._kidx = {k: None for k in self._keys}
        new_os = gym.spaces.Dict({
                k: v for k, v in os.items() if k in self._keys
        })
        for k, s in new_os.items():
            assert isinstance(s, (gym.spaces.Box, gym.spaces.MultiBinary, gym.spaces.MultiDiscrete, gym.spaces.Discrete))
            assert len(s.shape) <= 1  # Vectors and discrete only
            if len(s.shape):  # ignore discrete. If in fields, always included
                idx = np.array(kidx_pairs[k]) if (kidx_pairs is not None) else np.arange(s.shape[0])  # None value means all indexes
                new_os[k] = filter_gym_space(s, idx)
                self._kidx[k] = idx
        if getattr(env, "is_vector_env", False):
            from gym.vector.utils import batch_space
            self.single_observation_space = new_os
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else: self.observation_space = new_os


    def observation(self, observation):
        """Strip out some indices"""
        obs = {}
        for k, idx in self._kidx.items():
            obs[k] = observation[k] if (idx is None) else observation[k][..., idx]
        return obs




