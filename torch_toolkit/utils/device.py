__all__ = ['th_device', 'to_th', 'to_np']

from typing import Union, Iterable, Dict, Tuple, Any, Optional
# from numbers import Number
from dataclasses import is_dataclass, astuple
import torch as th
import numpy as np
Tensor = th.Tensor
Array = np.ndarray


def th_device(string: Union[str, th.device] = 'cuda') -> th.device:
    """Get torch device from string? (default to cuda)"""
    return th.device(string) if isinstance(string, str) else string


def to_th(buffer_: Union[Array, Tensor, Dict, Iterable, Any],
          device: Union[str, th.device] = 'cpu', dtype_override: Optional[th.dtype] = None):
    """Move to torch tensor(s) of given device"""
    if isinstance(buffer_, Array): return th.from_numpy(buffer_).to(device=device, dtype=dtype_override)
    elif isinstance(buffer_, Tensor): return buffer_.to(device=device, dtype=dtype_override)
    elif isinstance(buffer_, Iterable):
        if isinstance(buffer_, Tuple): return type(buffer_)((to_th(b, device) for b in buffer_))  # Immutable
        else:
            for i in range(len(buffer_)): buffer_[i] = to_th(buffer_[i], device)  # Potentially avoid copy
            return buffer_
    elif isinstance(buffer_, Dict):
        for k in buffer_.keys(): buffer_[k] = to_th(buffer_[k], device)
    elif is_dataclass(buffer_): return type(buffer_)((to_th(b, device) for b in astuple(buffer_)))
    else: return buffer_


def to_np(buffer_: Union[Array, Tensor, Dict, Iterable, Any]):
    """Move to numpy array(s)"""
    if isinstance(buffer_, Array): return Array
    elif isinstance(buffer_, Tensor): return buffer_.to('cpu').numpy()
    elif isinstance(buffer_, Iterable):
        if isinstance(buffer_, Tuple): return type(buffer_)((to_np(b) for b in buffer_))  # Immutable
        else:
            for i in range(len(buffer_)): buffer_[i] = to_np(buffer_[i])  # Potentially avoid copy
            return buffer_
    elif isinstance(buffer_, Dict):
        for k in buffer_.keys(): buffer_[k] = to_np(buffer_[k])
    elif is_dataclass(buffer_): return type(buffer_)((to_np(b) for b in astuple(buffer_)))
    else: return buffer_



