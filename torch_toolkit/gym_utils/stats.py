__all__ = ['EnvStatisticsRecorder', 'VectorizedTrajectoryRecorder']

from typing import Optional

import warnings
import numpy as np
import torch as th
import gym

from ..utils import NpBuffer, to_np, to_th


metrics = ['Length', 'Return', 'DiscountedReturn']  # Metrics to measure
metric_measures = {
    'Mean': np.nanmean,
    'Median': np.nanmedian,
    'Std': np.nanstd,
}
class EnvStatisticsRecorder:
    """Class that keeps track of a rolling window of environment statistics

    Args:
        B: Env batch size
        max_length: Size of window to maintain
        gamma: Discount factor used to calculate discounted return
    """
    def __init__(self, B: int = 1, max_length: int = 100, gamma: float = 1.):
        self._len = max_length
        self._gamma = gamma
        self.B = B
        self.maxlen = max_length
        self.clear()

    def step(self, r, d):
        r, d = to_np((r, d))
        self._length += 1
        self._return += r
        self._discounted_return += r * self._gamma
        self._discount *= self._gamma
        self._update_dones(d)

    def _update_dones(self, d):
        self.Length += self._length[d]
        self.Return += self._return[d]; self._return[d] = 0
        self.DiscountedReturn += self._discounted_return[d]
        self._length[d], self._return[d], self._discounted_return[d], self._discount[d] = 0, 0, 0, 1

    def clear(self):
        B, max_length = self.B, self.maxlen
        # Current episode stats
        self._length, self._return, self._discounted_return, self._discount = np.zeros(B), np.zeros(B), np.zeros(B), np.ones(B)
        # Stored stats
        self.Length, self.Return, self.DiscountedReturn = (NpBuffer(max_length) for _ in range(3))

    def get_statistics(self, nested: bool = True):
        """Get most recent episode statistics"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if nested:
                return {m: {m_m: m_mf(getattr(getattr(self, m), 'buffer')) for m_m, m_mf in metric_measures.items()} for m in metrics}
            else:
                return {f'charts/{m}_{m_m}': m_mf(getattr(getattr(self, m), 'buffer')) for m in metrics for m_m, m_mf in metric_measures.items()}

class VectorizedTrajectoryRecorder(gym.Wrapper):
    """Episode statistics recorder for vectorized environments"""
    _discount = 1.
    def __init__(self, env, max_deque_length: int = 100, discount: float = 1.):
        super().__init__(env)
        B = getattr(env, 'num_envs', 1)
        discount = getattr(env, 'discount', discount)  # Attempt to use environment discount if provided
        self.recorder = EnvStatisticsRecorder(B, max_deque_length, discount)

    def step(self, action):
        """Step environments and record statistics"""
        o, r, d, info = super().step(action)
        self._step(r, d)
        return o, r, d, info

    def _step(self, reward, done):
        """ Step AND take dones into account"""
        self.recorder.step(reward, done)

    def reset(self, discard_trajectories=False, **kwargs):
        """Reset all env instances and counters. Optionally, clear queue of completed trajectories"""
        o = super().reset(**kwargs)
        self._reset_stats()
        return o

    def _reset_stats(self):
        """Reset all counters to 0"""
        self.recorder.clear()

    def get_statistics(self, nested: bool = True):
        """Get most recent episode statistics"""
        return self.recorder.get_statistics(nested)