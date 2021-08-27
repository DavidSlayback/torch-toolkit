from typing import Optional, Union, Dict, Sequence
from numbers import Number
from functools import partial

from ..utils.device import th_device

import torch as th
Tensor = th.Tensor
import numpy as np
Array = np.ndarray


def _expand_to_t(x: Union[Array, Tensor], T: int):
    """Expand to buffer size"""
    if isinstance(x, Array): return np.tile(x, (T,) + x.shape[1:])
    else: return th.tile(x, (T,) + x.shape[1:])


def _convert_for_buffer(x: Union[Array, Tensor, Number], T:int,
                        to_tensor: bool = False, device: Union[str, th.device] = 'cpu') -> Union[Array, Tensor]:
    """Convert initial observation to appropriate type"""
    if isinstance(x, Number): x = np.array(x)[None, :]  # Convert to numpy if needed
    if x.dtype == np.float64: x = x.astype(np.float32)  # Downcast doubles
    if to_tensor: x = th.from_numpy(x).to(device).unsqueeze(0)  # Convert to torch if needed
    return _expand_to_t(x, T)

# class BaseRolloutBuffer:
#     """Base rollout buffer class
#
#     Store starting obs, reward, done
#
#     Initial arguments should include first observation and previous action (or null action)
#     """
#     o: Union[Array, Tensor, Number]  # Starting observation
#     a: Union[Array, Tensor, Number]  # Action taken
#     T: int  # Rollout length
#     r: Union[Array, Tensor, Number] = 0.  # Reward
#     d: Optional[Union[Array, Tensor, Number]] = True  # Done
#     i: int = 0  # Current index
#     is_tensor: bool = False  # Store as tensor or numpy arrays
#     device: Union[str, th.device] = 'cpu'  # Device if using
#     def __post_init__(self):
#         """Convert to arrays/tensors"""
#         b_fn = partial(_convert_for_buffer, to_tensor=self.is_tensor, device=self.device)
#         self.o = b_fn(self.o, self.T+1)  # o, next_o
#         self.a = b_fn(self.a, self.T+1)  # prev_a, a
#         self.r = b_fn(self.r, self.T+1)  # prev_r, r
#         self.d = b_fn(self.r, self.T+1)  # is_init, next_done
#
#     def __getitem__(self, item):
#         return




