__all__ = ['AttrDict', 'dotdict', 'arrdict', 'namedarrtuple']

from functools import wraps, partialmethod
import numpy as np
import torch as th
from .device import to_th, to_np, th_stack, np_stack

SCREEN_WIDTH = 119
SCREEN_HEIGHT = 200

class AttrDict(dict):
    """Simple Dict with attribute access"""
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


"""From Andy Jones! See https://andyljones.com/megastep/concepts.html for usage of dotdicts and arrdicts"""

class dotdict(dict):
    """dotdicts are dictionaries with additional support for attribute (dot) access of their elements.
    dotdicts have a lot of unusual but extremely useful behaviours, which are documented in :ref:`the dotdicts
    and arrdicts concept section <dotdicts>` .
    """

    def __dir__(self):
        return sorted(set(super().__dir__() + list(self.keys())))

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            try:
                gotten = [(k, getattr(v, key)) for k, v in self.items()]
            except AttributeError:
                raise AttributeError(
                    f"There is no member called '{key}' and one of the leaves has no attribute '{key}'") from None
            else:
                return type(self)(gotten)

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

    def copy(self):
        """Shallow-copy the dotdict"""
        return type(self)(**self)

    def pipe(self, f, *args, **kwargs):
        """Returns ``f(self, *args, **kwargs)`` .
        >>> d = dotdict(a=1, b=2)
        >>> d.pipe(list)
        ['a', 'b']
        Useful for method-chaining."""
        return f(self, *args, **kwargs)

    def map(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdict, returning a matching dotdict of the results.
        ``*args`` and  ``**kwargs`` are passed as extra arguments to each call.
        >>> d = dotdict(a=1, b=2)
        >>> d.map(int.__add__, 10)
        dotdict:
        a    11
        b    12
        Useful for method-chaining. Works equally well on trees of dotdicts.

        See :func:`mapping` for a functional version of this method."""
        return mapping(f)(self, *args, **kwargs)

    def starmap(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdicts one key at a time, returning a matching dotdict of the results.
        >>> d = dotdict(a=1, b=2)
        >>> d.starmap(int.__add__, d)
        dotdict:
        a    2
        b    4
        Useful for method-chaining. Works equally well on trees of dotdicts.

        See :func:`starmapping` for a functional version of this method."""
        return starmapping(f)(self, *args, **kwargs)


def treestr(t):
    """Stringifies a tree structure. These turn up all over the place in my code, so it's worth factoring out"""
    key_length = max(map(len, map(str, t.keys()))) if t.keys() else 0
    max_spaces = 4 + key_length
    val_length = SCREEN_WIDTH - max_spaces

    d = {}
    for k, v in t.items():
        if isinstance(v, dotdict):
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
    passed as extra arguments to ``f`` .
    >>> d = dotdict(a=1, b=2)
    >>> m = mapping(int.__add__)
    >>> m(d, 10)
    dotdict:
    a    11
    b    12

    Works equally well on trees of dotdicts, where ``f`` will be applied to the leaves of the tree.
    Can be used as a decorator.
    See :func:`dotdict.map` for an object-oriented version of this function.
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
    >>> d = dotdict(a=1, b=2)
    >>> m = starmapping(int.__add__)
    >>> m(d, d)
    dotdict:
    a    2
    b    4

    Works equally well on trees of dotdicts, where ``f`` will be applied to the leaves of the trees.
    Can be used as a decorator.
    See :func:`dotdict.starmap` for an object-oriented version of this function.
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


# TODO: Need to write a reduce really
def leaves(t):
    """Returns the leaves of a tree of dotdicts as a list"""
    if isinstance(t, dict):
        return [l for v in t.values() for l in leaves(v)]
    return [t]


def first_value(t):
    if isinstance(t, dict):
        return first_value(next(iter(t)))
    return t


"""Arrdicts. Dotdicts with index access"""

def _is_valid_field(x):
    """Valid fields are strings of collections thereof"""
    return (isinstance(x, str) or (isinstance(x, tuple) and all(isinstance(xx, str) for xx in x)))


def _arrdict_factory():
    # This is done with a factory because I am a lazy man and I didn't fancy defining all the binary ops on
    # the arrdict manually.

    class _arrdict_base(dotdict):
        """An arrdict is an :class:`~rebar.dotdict.dotdict` with extra support for array and tensor values.
        arrdicts have a lot of unusual but extremely useful behaviours, which are documented in :ref:`the dotdicts
        and arrdicts concept section <dotdicts>` .
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, x):
            """Get all fields at index x. If x is str, get the field 'x'"""
            if isinstance(x, str):
                return super().__getitem__(x)
            else:
                return type(self)({k: v[x] for k, v in self.items()})

        def __setitem__(self, x, y):
            """If x is string, attempt to set field
            If y is another arrdict, update our keys at location x with fields from y
            """
            # Valid keys to stick in an arrdict are strings and tuples of strings.
            # Anything else could plausibly be a tensor index.
            if _is_valid_field(x):
                super().__setitem__(x, y)
            elif isinstance(y, type(self)):
                for k in self:
                    self[k][x] = y[k]
            else:
                raise ValueError('Setting items must be done with a string key or by passing an arrdict')

        def __binary_op__(self, name, rhs):
            """If rhs is dict instance, do op (see below). Otherwise, apply function"""
            if isinstance(rhs, dict):
                return self.starmap(name, rhs)
            else:
                return super().__getattr__(name)(rhs)

    # Add binary methods
    # https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    binaries = [
        'lt', 'le', 'eq', 'ne', 'ge', 'gt',
        'add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'or',
        'xor',
        'radd', 'rsub', 'rmul', 'rmatmul', 'rtruediv', 'rfloordiv', 'rmod', 'rdivmod', 'rpow', 'rand', 'lshift',
        'rshift', 'ror', 'rxor']
    methods = {f'__{name}__': partialmethod(_arrdict_base.__binary_op__, f'__{name}__') for name in binaries}

    methods['__doc__'] = _arrdict_base.__doc__

    return type('arrdict', (_arrdict_base,), methods)


arrdict = _arrdict_factory()
def namedarrtuple(name='AnonymousNamedArrTuple', fields=()):
    """arrdict converted to work like a namedarraytuple

    Args:
        name: Name of tuple class
        fields: Set of fields
    """
    def __init__(self, *args, **kwargs):
        super(arrdict, self).__init__(*args, **kwargs)
        if set(fields) != set(self):
            raise KeyError(f'This NamedArrTuple subclass must be created with exactly the fields {fields}')

    def __setitem__(self, x, y):
        if _is_valid_field(x) and (x not in fields):
            raise KeyError(f'Key "{x}" is not in this immutable NamedArrTuple, and so cannot be added')
        super(arrdict, self).__setitem__(x, y)

    return type(name, (arrdict,), {'__init__': __init__, '__setitem__': __setitem__})


@mapping
def torchify(a):
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
        dtype = th.bool
    else:
        raise ValueError(f'Can\'t handle {type(a)}')
    return th.as_tensor(np.array(a), dtype=dtype)


@mapping
def numpyify(tensors):
    """Converts an array or a dict of tensors to numpy arrays.
    """
    if isinstance(tensors, tuple):
        return tuple(numpyify(t) for t in tensors)
    if isinstance(tensors, th.Tensor):
        return tensors.clone().detach().cpu().numpy()
    if hasattr(tensors, 'numpyify'):
        return tensors.numpyify()
    return tensors


def stack(x, *args, **kwargs):
    """Stacks a sequence of arrays, tensors or dicts thereof.
    For example,
    >>> d = arrdict(a=1, b=np.array([1, 2]))
    >>> stack([d, d, d])
    arrdict:
    a    ndarray((3,), int64)
    b    ndarray((3, 2), int64)
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
    For example,
    >>> d = arrdict(a=1, b=np.array([1, 2]))
    >>> cat([d, d, d])
    arrdict:
    a    ndarray((3,), int64)
    b    ndarray((6,), int64)
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
    """Create arrdict from dict"""
    if isinstance(t, dict):
        return arrdict({k: from_dicts(v) for k, v in t.items()})
    return t


def to_dicts(t):
    """Create dict from arrdict"""
    if isinstance(t, dict):
        return {k: to_dicts(v) for k, v in t.items()}
    return t


