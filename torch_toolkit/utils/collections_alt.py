__all__ = ['ArrayDataclassMixin', 'ArrayDict']

from typing import NamedTuple, Dict, Union, Optional
from functools import wraps

import torch as th
import numpy as np
Tensor = th.Tensor
Array = np.ndarray
XArray = Union[Tensor, Array]
MaybeXArray = Optional[XArray]
SCREEN_WIDTH = 119
SCREEN_HEIGHT = 200

import sys
from collections import namedtuple, OrderedDict
from inspect import Signature as Sig, Parameter as Param
import string

RESERVED_NAMES = ("get", "items")

class ArrayDataclassMixin:
    """Mixin for dataclasses that allows index access
    """
    def __iter__(self):
        """Ensure we have an iter for easy use of __getitem__"""
        return iter(self.__tuple__)

    def __getitem__(self, loc):
        """Get all fields at index"""
        return type(self)(*(v[loc] for v in self))

    def __setitem__(self, loc, value):
        """Set fields at index"""
        if isinstance(value, (tuple, list)) or (type(value) == type(self)):  # Update with sequence
            for i, k in enumerate(self): k[loc] = value[i]
        elif isinstance(value, dict):  # Update with dict
            for k in self.__slots__: getattr(self, k)[loc] = value[k]



class NamedArrayTupleMixin:
    """NamedTuple mixin for array access. Init with desired buffers"""
    def __getitem__(self, loc):
        """Get attribute or items at index"""
        if isinstance(loc, str): return getattr(self, loc)  # field access
        else: return type(self)(*(None if s is None else s[loc] for s in self))  # index access

    def __setitem__(self, loc, value):
        """Set items at index"""
        if not isinstance(value, tuple) and getattr(value, '_fields', None) == self._fields: value = tuple(None if s is None else value for s in self)
        for j, (s,v) in enumerate(zip(self, value)):
            if s is not None or v is not None: s[loc] = v

    def __contains__(self, item):
        """Do we have this field"""
        return item in self._fields

    def get(self, loc):
        """Typical tuple __getitem__ access"""
        return super().__getitem__(loc)

    def items(self):
        """Return k, v pairs as in dict"""
        for k, v in zip(self._fields, self): yield k, v


def treestr(t):
    """Stringifies a tree structure. These turn up all over the place in my code, so it's worth factoring out"""
    key_length = max(map(len, map(str, t.keys()))) if t.keys() else 0
    max_spaces = 4 + key_length
    val_length = SCREEN_WIDTH - max_spaces

    d = {}
    for k, v in t.items():
        if isinstance(v, ArrayDict):
            d[k] = str(v)
        elif isinstance(v, (list, set, dict)):
            d[k] = f'{type(v).__name__}({len(v)},)'
        elif hasattr(v, 'shape') and hasattr(v, 'dtype'):
            d[k] = f'{type(v).__name__}({tuple(v.shape)}, {v.dtype})'
        elif hasattr(v, 'shape'):
            d[k] = f'{type(v).__name__}({tuple(v.shape)})'
        else:
            lines = str(v).splitlines()
            if (len(lines) > 1) or (len(lines[0]) > val_length):
                d[k] = lines[0][:val_length] + ' ...'
            else:
                d[k] = lines[0]

    s = [f'{type(t).__name__}:']
    for k, v in d.items():
        lines = v.splitlines() or ['']
        s.append(str(k) + ' ' * (max_spaces - len(str(k))) + lines[0])
        for l in lines[1:]:
            s.append(' ' * max_spaces + l)
        if len(s) >= SCREEN_HEIGHT - 1:
            s.append('...')
            break

    return '\n'.join(s)

def mapping(f):
    """Wraps ``f`` so that when called on a dotdict, ``f`` instead gets called on the dotdict's values
    and a dotdict of the results is returned. Extra ``*args`` and ``**kwargs`` passed to the wrapper are
    passed as extra arguments to ``f``
    """
    @wraps(f)
    def g(x, *args, **kwargs):
        if isinstance(x, dict):
            return type(x)([(k, g(v, *args, **kwargs)) for k, v in x.items()])
        if isinstance(f, str):
            return getattr(x, f)(*args, **kwargs)
        return f(x, *args, **kwargs)
    return g


def starmapping(f):
    """Wraps ``f`` so that when called on a sequence of dotdicts, ``f`` instead gets called on the dotdict's values
    and a dotdict of the results is returned.
    """
    @wraps(f)
    def g(x, *args, **kwargs):
        if isinstance(x, dict):
            return type(x)([(k, g(x[k], *(a[k] for a in args))) for k in x])
        if isinstance(f, str):
            return getattr(x, f)(*args)
        else:
            return f(x, *args)
    return g

def _is_valid_field(x):
    """Valid fields are strings of collections thereof"""
    return (isinstance(x, str) or (isinstance(x, tuple) and all(isinstance(xx, str) for xx in x)))


class ArrayDict(dict):
    """Dict with array indexing for its fields"""
    def __call__(self, *args, **kwargs):
        """Return new ArrayDict with same elements. Done this way in case subfields are also ArrayDicts"""
        return type(self)([(k, v(*args, **kwargs)) for k, v in self.items()])

    def __getitem__(self, loc):
        """Index access"""
        if isinstance(loc, str): return dict.__getitem__(self, loc)  # Standard dictionary access
        else: return type(self)({k: v[loc] for k, v in self.items()})  # Index access

    def __getattr__(self, item):
        """Dot access"""
        if item in self: return self[item]  # Dot access for dict
        else:
            try:
                gotten = [(k, getattr(v, item)) for k, v in self.items()]  # Check all children (which themselves could check)
            except AttributeError:
                raise AttributeError(
                    f"There is no member called '{item}' and one of the leaves has no attribute '{item}'") from None
            else:
                return type(self)(gotten)

    def __setitem__(self, key, value):
        """Index or dot access"""
        if _is_valid_field(key): dict.__setitem__(self, key, value)  # Set field
        elif isinstance(value, type(self)):  # Matching ArrayDict, update
            for k in self: self[k][key] = value[k]
        else: raise ValueError('Cannot set')

    def __str__(self):
        """Stringify"""
        return treestr(self)

    def __repr__(self):
        """Stringify"""
        return self.__str__()

    def __getstate__(self):
        """Serialize (dict)"""
        return self

    def __setstate__(self, state):
        """Deserialize (dict)"""
        self.update(state)

    def copy(self):
        """Shallow-copy"""
        return type(self)(**self)

    def pipe(self, f, *args, **kwargs):
        """Returns ``f(self, *args, **kwargs)``.
        e.g., d.pipe(list) returns a list of elements"""
        return f(self, *args, **kwargs)

    def map(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdict, returning a matching dotdict of the results.
        ``*args`` and  ``**kwargs`` are passed as extra arguments to each call.
        """
        return mapping(f)(self, *args, **kwargs)

    def starmap(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdicts one key at a time, returning a matching dotdict of the results."""
        return starmapping(f)(self, *args, **kwargs)


@mapping
def torchify(a, device: Optional[Union[th.device, str]] = 'cpu', override_bool: bool = False):
    """Converts an array or a dict of numpy arrays to CPU tensors.
    If you'd like CUDA tensors, follow the tensor-ification ``.cuda()`` ; the attribute delegation
    built into :class:`~rebar.dotdict.dotdict` s will do the rest.

    Floats get mapped to 32-bit PyTorch floats; ints get mapped to 64-bit PyTorch ints. This is usually what you want in
    machine learning work.
    """
    if hasattr(a, 'torchify'):
        return a.torchify()
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        dtype = th.float32
    elif np.issubdtype(a.dtype, np.integer):
        dtype = th.int64
    elif np.issubdtype(a.dtype, np.bool_):
        dtype = th.bool if not override_bool else th.float32
    else:
        raise ValueError(f'Can\'t handle {type(a)}')
    return th.as_tensor(np.array(a), dtype=dtype, device=device)


@mapping
def numpyify(tensors):
    """Converts an array or a dict of tensors to numpy arrays."""
    if isinstance(tensors, tuple):
        return tuple(numpyify(t) for t in tensors)
    if isinstance(tensors, th.Tensor):
        return tensors.clone().detach().cpu().numpy()
    if hasattr(tensors, 'numpyify'):
        return tensors.numpyify()
    return tensors


def stack(x, *args, **kwargs):
    """Stacks a sequence of arrays, tensors or dicts thereof.
    Any ``*args`` or ``**kwargs`` will be forwarded to the ``np.stack`` or ``torch.stack`` call.
    Python scalars are converted to numpy scalars, so - as in the example above - stacking floats will
    get you a 1D array.
    """
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: stack([y[k] for y in x], *args, **kwargs) for k in ks})
    if isinstance(x[0], th.Tensor):
        return th.stack(x, *args, **kwargs)
    if isinstance(x[0], np.ndarray):
        return np.stack(x, *args, **kwargs)
    if np.isscalar(x[0]):
        return np.array(x, *args, **kwargs)
    raise ValueError(f'Can\'t stack {type(x[0])}')


def cat(x, *args, **kwargs):
    """Concatenates a sequence of arrays, tensors or dicts thereof.
    Any ``*args`` or ``**kwargs`` will be forwarded to the ``np.concatenate`` or ``torch.cat`` call.
    Python scalars are converted to numpy scalars, so - as in the example above - concatenating floats will
    get you a 1D array.
    """
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: cat([y[k] for y in x], *args, **kwargs) for k in ks})
    if isinstance(x[0], th.Tensor):
        return th.cat(x, *args, **kwargs)
    if isinstance(x[0], np.ndarray):
        return np.concatenate(x, *args, **kwargs)
    if np.isscalar(x[0]):
        return np.array(x)
    raise ValueError(f'Can\'t cat {type(x[0])}')


@mapping
def clone(t):
    """Copy/clone elements of dotdict/arrdict"""
    if hasattr(t, 'clone'):
        return t.clone()
    if hasattr(t, 'copy'):
        return t.copy()
    return t


def from_dicts(t):
    """Create ArrayDict from dict"""
    if isinstance(t, dict):
        return ArrayDict({k: from_dicts(v) for k, v in t.items()})
    return t


def to_dicts(t):
    """Create dict of [dicts] from ArrayDict"""
    if isinstance(t, dict):
        return {k: to_dicts(v) for k, v in t.items()}
    return t