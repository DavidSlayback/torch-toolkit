import dataclasses
from typing import Dict

import torch

from ..typing import *

# Types to index into thing
IndexType = Union[slice, int, np.ndarray, Sequence[int]]

GTensor = TypeVar('GTensor', Tensor, TensorDict)

class BufferDict(dict):
    """Dictionary with array indexing. Keys must be strings only"""

    def __dir__(self):
        return sorted(set(super().__dir__() + list(self.keys())))

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            try:
                return type(self)([(k, getattr(v, key)) for k, v in self.items()])
            except AttributeError:
                raise AttributeError(
                    f"There is no member called '{key}' and one of the leaves has no attribute '{key}'") from None

    def __call__(self, *args, **kwargs):
        return type(self)([(k, v(*args, **kwargs)) for k, v in self.items()])

    def __str__(self):
        return treestr(self)

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __getitem__(self, x: Union[str, IndexType]):
        """If string index, be a dict. Otherwise be an array"""
        if isinstance(x, str): return super().__getitem__(x)
        else: return type(self)({k: v[x] for k, v in self.items()})

    def __setitem__(self, k: Union[str, IndexType], v):
        """If string index, be a dict. Otherwise update from comparable BufferDict"""
        if isinstance(k, str): super().__setitem__(k, v)  # Set string key
        elif isinstance(v, type(self)):
            for k in self: self[k][v] = v[k]

    def __setattr__(self, key, value):
        raise ValueError('Setting by attribute is not allowed; set by key instead')

    def __binary_op__(self, name, rhs):
        if isinstance(rhs, dict):
            return self.starmap(name, rhs)
        else:
            return super().__getattr__(name)(rhs)

    ...

def buffer_from_example(obs: Union[XArray, XArrayDict],
                        act: Union[XArray, XArrayDict],
                        T: int,
                        B: Optional[int] = None,
                        buffer_type: Union[Tensor, np.ndarray] = Tensor,
                        buffer_device: Optional[Union[str, torch.device]] = None,
                        **kwargs) -> BufferDict:
    """Create buffer from examples

    Args:
    """
    ...