__all__ = ['ThSpace', 'ThDiscrete']

from typing import Union
from gym.spaces import Box, Discrete, Space
import numpy as np
import torch as th
from ... import th_device

"""Numpy-torch converstions"""
NP_TO_TH = {
    np.float64: th.float64,
    np.float32: th.float32,
    np.float16: th.float16,
    np.int64: th.int64,
    np.int32: th.int64,  # Int32 tensors are pretty useless
    np.uint8: th.uint8,
    np.bool: th.bool,
}


class ThSpace(Space):
    """Torch version of gym space. Use global torch rng"""
    def __init__(self, shape=None, dtype=None, device: Union[str, th.device] = 'cpu'):
        super().__init__(shape, dtype)
        self.dtype = NP_TO_TH[self.dtype] if dtype is not None else None
        self.device = th_device(device)


class ThDiscrete(Discrete, ThSpace):
    def sample(self):
        return th.randint(self.n, (), device=self.device, dtype=self.dtype)


# class ThBox(Box, ThSpace):
#     def __init__(self, *args, device: Union[str, th.device] = 'cpu', **kwargs):
#         Box.__init__(self, *args, **kwargs)
#         self.dtype = NP_TO_TH[self.dtype] if self.dtype is not None else None
#         self.device = th_device(device)
#         self.low, self.high, self.bounded_below, self.bounded_above = to_th((self.low, self.high, self.bounded_below, self.bounded_above), device)
#     def sample(self):
#         ...