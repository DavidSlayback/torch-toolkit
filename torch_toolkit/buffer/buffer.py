__all__ = ['AgnosticRolloutBuffer']

from typing import Optional, Union, Dict, Iterable, Any, Tuple
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


# def _expand_to_t(x: Union[Array, Tensor], T: int):
#     """Expand to buffer size"""
#     if isinstance(x, Array): return np.tile(x, (T,) + x.shape[1:])
#     else: return th.tile(x, (T,) + x.shape[1:])
#
#
# def _convert_for_buffer(x: Union[Array, Tensor, Number], T:int,
#                         to_tensor: bool = False, device: Union[str, th.device] = 'cpu') -> Union[Array, Tensor]:
#     """Convert initial observation to appropriate type"""
#     if isinstance(x, Number): x = np.array(x)[None, :]  # Convert to numpy if needed
#     if x.dtype == np.float64: x = x.astype(np.float32)  # Downcast doubles
#     if to_tensor: x = th.from_numpy(x).to(device).unsqueeze(0)  # Convert to torch if needed
#     return _expand_to_t(x, T)


class AgnosticRolloutBuffer:
    """Agnostic buffer class that defines its fields on first step

    Args:
        T: Number of timesteps to store
        device: One of [None,cpu,cuda]. If None, store inputs as numpy. Otherwise, tensors of appropriate device
    """
    def __init__(self, T: int, device: Optional[Union[str, th.device]] = None):
        self._T = T
        self._B = 0
        self._keys = set()
        self._buffer_keys = set()
        self._i = 0
        self._as_tensor = device is not None
        self.device = th_device(device)
        self.memory = {}
        self.convert = partial(to_th, device=self.device) if self._as_tensor else to_np

    def _initialize(self, **kwargs: ArrDict) -> None:
        """Called first time an experience is added. Add buffers for each given key in dictionary.

        Does some special logic to share memory for instances where we might have obs, next_obs keys

        Args:
            kwargs: k, v pairs of argument names and first inputs
        """
        for k, v in kwargs.items():
            if k in self._keys: continue  # Ignore keys we already have
            dtype = torch_type_to_np(v.dtype)  # Get dtype so we can downcast float64 to float32 if needed
            if isinstance(dtype, (np.float64, np.bool)): dtype = np.float32
            if '_' in k:
                spl = k.split('_')
                if len(spl) > 1 and spl[0] in B_RESERVED:
                    b_shape = (self._T+1,) + tuple(v.shape)  # Get general shape
                    base_buffer = self.convert(np.zeros(b_shape, dtype=dtype))  # Create and convert buffer
                    basekey = '_'.join(spl[1:])
                    bkey = basekey + '_buffer'  # Easy access to full buffer
                    self._buffer_keys.add(bkey)
                    self.memory[bkey] = base_buffer
                    if spl[0] == 'next':
                        self.memory[k] = base_buffer[1:]; self.memory[basekey] = base_buffer[:-1]
                        self.memory[basekey][self._i] = self.convert(v)
                    else:
                        self.memory[basekey] = base_buffer[1:]; self.memory[k] = base_buffer[:-1]
                        self.memory[k][self._i] = self.convert(v)
                    self._keys.add(basekey)
            else:
                b_shape = (self._T,) + tuple(v.shape)  # Get general shape
                base_buffer = self.convert(np.zeros(b_shape, dtype=dtype))  # Create and convert buffer
                base_buffer[self._i] = self.convert(v)
                self.memory[k] = base_buffer  # Create buffer for key
            self._keys.add(k)
        self._B = b_shape[0]  # Batch dimension

    def add(self, **kwargs: ArrDict) -> None:
        """Add an experience"""
        if len(self._keys) == 0:
            self.add = self._add  # Remove conditional
            return self._initialize(**kwargs)  # Init with args if not done
        for k, v in kwargs.items(): self.memory[k][self._i] = self.convert(v)  # Store
        self._i = self._i + 1  # Update index

    def _add(self, **kwargs: ArrDict) -> None:
        """Replace above"""
        for k, v in kwargs.items(): self.memory[k][self._i] = self.convert(v)  # Store
        self._i = self._i + 1  # Update index

    def finish(self, **kwargs: ArrDict) -> None:
        """Finish with boostraps"""
        assert self._i == self._T
        for k, v in kwargs.items():
            self.memory['next_'+k][self._i - 1] = v

    def add_extra(self, **kwargs) -> None:
        """Add extras computed externally (e.g., advantages and returns in policy-gradient algorithms)."""
        for k, v in kwargs.items():
            self.memory[k] = v
            self._keys.add(k)  # Add to our set of keys

    def get_all(self) -> ArrDict:
        """Get all experiences"""
        assert self._i == self._T
        return self.memory

    def mb_sample(self, n_batch: int, shuffle: bool = True, save_temporal_correlation: bool = True):
        """Sample minibatches from the buffer

        Args:
            n_batch: Number of minibatches to sample
            shuffle: Whether to randomize samples
            save_temporal_correlation: If true, only randomize over batch dimension
        """
        assert self._i == self._T
        T, B = self._T, self._B
        if save_temporal_correlation:
            idx = np.arange(B)
            if shuffle: np.random.shuffle(idx)
            for bidx in np.array_split(idx, n_batch):
                yield self[:, bidx]
        else:
            idx = np.arange(int(T*B))
            if shuffle: np.random.shuffle(idx)
            for bidx in np.array_split(idx, n_batch):
                yield self[np.unravel_index(bidx, (T, B))]

    def reset(self):
        """Reset buffer."""
        self._i = 0
        for k in self._buffer_keys: self.memory[k][0] = self.memory[k][-1]  # Rollover buffers

    def get_last(self, keys: Iterable[str]):
        """Return requested keys at current idx. Useful to get items converted to tensor"""
        return self._get_some(keys, self._i)

    def __getitem__(self, item) -> ArrDict:
        """Access experiences at some index"""
        return {k: self.memory[k][item] for k in self._keys} # Ignore buffer keys
        # return {k: v[item] for k, v in self.memory.items()}

    def _get_some(self, keys: Iterable[str], item) -> ArrDict:
        """Get only a subset of keys. No error checking here"""
        return {k: self.memory[k][item] for k in keys}

    def __len__(self):
        return self._i





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




