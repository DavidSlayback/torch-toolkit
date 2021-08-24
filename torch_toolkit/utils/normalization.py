__all__ = ['RunningMeanStd', 'RunningMeanStdTorch']

from typing import Tuple, Union

import numpy as np
import torch as th
from .device import th_device


class RunningMeanStd:
    __slots__ = ('mean', 'var', 'count')
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Args:
            epsilon: helps with arithmetic issues
            shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RunningMeanStdTorch:
    __slots__ = ('mean', 'var', 'count', 'device')
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device: Union[str, th.device] = 'cpu'):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Args:
            epsilon: helps with arithmetic issues
            shape: the shape of the data stream's output
            device: torch device to store on
        """
        self.mean = th.zeros(shape, dtype=th.float64, device=th_device(device), requires_grad=False)
        self.var = th.ones(shape, dtype=th.float64, device=th_device(device), requires_grad=False)
        self.count = epsilon

    def update(self, arr: th.Tensor) -> None:
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: th.Tensor, batch_var: th.Tensor, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count