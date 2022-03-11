__all__ = ['th_device', 'to_th', 'to_np', 'th_stack', 'np_stack', 'torch_type_to_np', 'np_type_to_torch']

from typing import Union, Iterable, Dict, Tuple, Any, Optional, List
from numbers import Number
from dataclasses import is_dataclass, astuple
import torch as th
import numpy as np
Tensor = th.Tensor
Array = np.ndarray


def th_device(string: Optional[Union[str, th.device]] = 'cuda') -> Optional[th.device]:
    """Get torch device from string? (default to cuda). Return None"""
    return th.device(string) if isinstance(string, str) else string


def to_th(buffer_: Union[Array, Tensor, Dict, Iterable, Any],
          device: Union[str, th.device] = 'cpu', dtype_override: Optional[th.dtype] = None, inplace: bool = False):
    """Move to torch tensor(s) of given device"""
    if isinstance(buffer_, Array): return th.from_numpy(buffer_).to(device=device, dtype=dtype_override)
    elif isinstance(buffer_, Tensor): return buffer_.to(device=device, dtype=dtype_override)
    elif isinstance(buffer_, Dict):
        return type(buffer_)(**{k: to_th(v, device, dtype_override) for k, v in buffer_.items()})
    elif isinstance(buffer_, Iterable):
        return type(buffer_)((to_th(b, device, dtype_override) for b in buffer_))
    #     if isinstance(buffer_, Tuple): return type(buffer_)((to_th(b, device) for b in buffer_))  # Immutable
    #     elif hasattr(buffer_, '__slots__'):
    #         for k in buffer_.__slots__: setattr(buffer_, k, to_th(getattr(buffer_, k))) # Hack to do my slotted dataclasses inplace
    #         return buffer_
    #     else:
    #         for i in range(len(buffer_)): buffer_[i] = to_th(buffer_[i], device)  # Potentially avoid copy
    #         return buffer_
        # for k in buffer_.keys(): buffer_[k] = to_th(buffer_[k], device)
    elif is_dataclass(buffer_): return type(buffer_)((to_th(b, device) for b in astuple(buffer_)))
    else: return buffer_


def to_np(buffer_: Union[Array, Tensor, Dict, Iterable, Any]):
    """Move to numpy array(s)."""
    if isinstance(buffer_, Array): return buffer_
    elif isinstance(buffer_, Tensor): return buffer_.cpu().numpy()
    elif isinstance(buffer_, Iterable):
        return type(buffer_)((to_np(b) for b in buffer_))
        # if isinstance(buffer_, Tuple): return type(buffer_)((to_np(b) for b in buffer_))  # Immutable
        # elif hasattr(buffer_, '__slots__'):
        #     for k in buffer_.__slots__: setattr(buffer_, k, to_np(getattr(buffer_, k))) # Hack to do my slotted dataclasses inplace
        #     return buffer_
        # else:
        #     for i in range(len(buffer_)): buffer_[i] = to_np(buffer_[i])  # Potentially avoid copy
        #     return buffer_
    elif isinstance(buffer_, Dict):
        return type(buffer_)(**{k: to_np(v) for k, v in buffer_.items()})
        # for k in buffer_.keys(): buffer_[k] = to_np(buffer_[k])
    elif is_dataclass(buffer_): return type(buffer_)((to_np(b) for b in astuple(buffer_)))
    else: return buffer_


def th_stack(buffer_: Union[Tuple, List], device: Union[str, th.device] = 'cpu') -> Tensor:
    """Stack iterable along 0-th dimension as torch tensor"""
    example = buffer_[0]  # Get example
    if isinstance(example, Tensor): return th.stack(buffer_).to(device)
    elif isinstance(example, Array): return to_th(np.stack(buffer_), device)
    elif isinstance(example, Number): return th.tensor(buffer_, device=device)
    else: raise ValueError("Input cannot be converted to torch stack")


def np_stack(buffer_: Union[Tuple, List]) -> Array:
    """Stack iterable along 0-th dimension as numpy array"""
    example = buffer_[0]
    if isinstance(example, Tensor): return th.stack(buffer_).cpu().numpy()
    elif isinstance(example, Array): return np.stack(buffer_)
    elif isinstance(example, Number): return np.array(buffer_)
    else: raise ValueError("Input cannot be converted to numpy stack")


def torch_type_to_np(d: Union[th.dtype, np.dtype]) -> np.dtype:
    """Return numpy dtype corresponding to torch dtype"""
    return th.empty((), dtype=d).numpy().dtype if isinstance(d, th.dtype) else d


def np_type_to_torch(d: Union[th.dtype, np.dtype]) -> th.dtype:
    """Return torch dtype corresonding to numpy dtype"""
    return th.from_numpy(np.empty((), dtype=d)).dtype if isinstance(d, np.dtype) else d



