from typing import Optional, Union, Dict, Iterable, Any, Tuple, Set
from functools import partial

from ..utils.device import th_device, to_th, to_np, torch_type_to_np

import torch as th
Tensor = th.Tensor
import numpy as np
Array = np.ndarray
Arr = Union[Array, Tensor]
ArrDict = Dict[str, Arr]

# Modifiers for keyword arguments
B_RESERVED = [
    'next',  # e.g., obs, next_obs or value, next_value
    'prev',  # e.g., prev_action, action
]

class BufferInterface:
    """Base buffer class interface"""
    def add(self, ):
        """Add k,v pairs to buffer at current index"""
        raise NotImplementedError

    def mb_sample(self, n_batch: int, shuffle: bool = True, save_temporal_correlation: bool = True):
        """Sample n_batch minibatches from buffer"""
        raise NotImplementedError

    def all(self):
        """Return full rollout"""
        raise NotImplementedError

    def full(self):
        """Is buffer full?"""
        return self._i == self._T

    def reset(self) -> None:
        """Reset buffer"""
        raise NotImplementedError

    def __len__(self) -> int: return self._i




def buffer_from_examples(key_examples: ArrDict, T: int,
                         input_keys: Iterable[str] = ('o', 'a'),
                         boostrap_keys: Iterable[str] = ('v')
                         ) -> BufferInterface:
    ...



class RolloutBuffer:
    _base_keys = ('o', 'r', 'd')  # Obs, reward, done
    """More structured rollout buffer class

    Args:
        T: Rollout length
        input_keys: Set of keys used for input, save "previous"
        bootstrap_keys: Set of keys used for boostrap, save "next"
        extra_keys: Externally calculated keys that will be added all at once (e.g., GAE, discounted returns)
        device: One of [None,cpu,cuda]. If None, store inputs as numpy. Otherwise, tensors of appropriate device
        step_device: One of [None,cpu,cuda]. If None, store inputs as numpy. Otherwise, tensors of appropriate device 
    """
    def __init__(self, T: int,
                 input_keys: Iterable[str] = ('o', 'a', 'd'),
                 bootstrap_keys: Iterable[str] = ('v'),
                 extra_keys: Iterable[str] = ('adv', 'ret'),
                 storage_device: Optional[Union[str, th.device]] = None,
                 step_device: Union[str, th.device] = 'cpu',
                 grad_device: Union[str, th.device] = 'cuda'):
        self._T = T
        self.memory = {}
        self._input_keys: Set[str] = set(input_keys)
        self._bootstrap_keys: Set[str] = set(bootstrap_keys)
        self._keys = set(extra_keys)  # Otherwise empty until initialization
        self._i = 0
        self._as_tensor = storage_device is not None
        self.device = th_device(storage_device)
        self.convert = partial(to_th, device=self.device) if self._as_tensor else to_np
        self.convert_step = partial(to_th, device=th_device(step_device))
        self.convert_grad = partial(to_th, device=th_device(grad_device))

    def initialize(self, key_examples: ArrDict) -> ArrDict:
        """Initialize using key_example pairs"""
        self._keys |= set(list(key_examples.keys()))  # Get keys from dict
        self._bootstrap_keys &= self._keys  # Only bootstrap keys we've seen
        self._input_keys &= self._keys  # Same for input
        self._buffer_keys = self._bootstrap_keys | self._input_keys  # Set of keys needing extra storage
        for k, v in key_examples.items():
            dtype = torch_type_to_np(v.dtype)  # Get dtype so we can downcast float64 to float32 if needed
            if isinstance(dtype, (np.float64, np.bool)): dtype = np.float32  # Booleans and doubles to floats
            if k in self._buffer_keys:
                b_shape = (self._T+1,) + tuple(v.shape)  # Get general shape
                base_buffer = self.convert(np.zeros(b_shape, dtype=dtype))  # Create and convert buffer
                # Save example as first element of buffer (e.g., store action for use as "prev_action")
                if k in self._input_keys: base_buffer[0] = self.convert(v)
                self.memory[k] = base_buffer[:-1]  # Ignore last element in PPO sampling
                self.memory[k + '_b'] = base_buffer  # Buffer key
            else:
                b_shape = (self._T,) + tuple(v.shape)  # Get general shape
                self.memory[k] = self.convert(np.zeros(b_shape, dtype=dtype))  # Create and convert buffer
            self._buffer_keys = set((k +'_b' for k in self._buffer_keys))
        return self.get_next_input()

    def get_next_input(self):
        """Return inputs as given by defined keys"""
        return self.convert_step({k: self.memory[k+'_b'][self._i] for k in self._input_keys})

    def add(self, **kwargs):
        """Add next step"""
        assert self._i < self._T
        for k, v in kwargs.items(): self.memory[k][self._i] = self.convert(v)
        self._i += 1

    def finish(self, **kwargs) -> ArrDict:
        """Add boostraps to buffers"""
        assert self._i == self._T
        for k, v in kwargs.items():
            self.memory[k+'_b'][self._i] = self.convert(v)

    def add_extras(self, **kwargs):
        """Add externally-calculated extras"""
        for k, v in kwargs.items():
            self.memory[k] = self.convert(v)

    def __getitem__(self, item):
        return {k: self.memory[k][item] for k in self._keys}  # Ignore buffer keys

