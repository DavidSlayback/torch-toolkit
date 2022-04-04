__all__ = ['th_ravel_multi_index', 'th_unravel_index', 'batched_index', 'infer_leading_dims']

from typing import Tuple
import torch as th
from ..typing import Tensor


# See https://github.com/francois-rozet/torchist/blob/5a53be59493e1e5ccd8a9e261796edba5c40b733/torchist/__init__.py#L18
def th_ravel_multi_index(coords: Tensor, shape: Tuple[int]) -> Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.
    This is a `torch` implementation of `numpy.ravel_multi_index`.
    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.
    Returns:
        The raveled indices, (*,).
    """

    coef = coords.new_tensor(shape[1:] + (1,))
    coef = coef.flipud().cumprod(0).flipud()
    if coords.is_cuda and not coords.is_floating_point():
        return (coords * coef).sum(dim=-1)
    else:
        return coords @ coef


def th_unravel_index(indices: Tensor, shape: Tuple[int]) -> Tensor:
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.
    Returns:
        The unraveled coordinates, (*, D).
    """
    coords = []
    for dim in reversed(shape):
        coords.append(indices % dim)
        indices = indices // dim
    coords = th.stack(coords[::-1], dim=-1)
    return coords


def batched_index(idx: Tensor, t: Tensor) -> Tensor:
    """Return contents of t at n-D array idx. Leading dim of t must match dims of idx"""
    dim = len(idx.shape)
    assert idx.shape == t.shape[:dim]
    num = idx.numel()
    t_flat = t.view((num,) + t.shape[dim:])
    s_flat = t_flat[th.arange(num, device=t.device), idx.view(-1)]
    return s_flat.view(t.shape[:dim] + t.shape[dim + 1:])


def infer_leading_dims(tensor: Tensor, dim: int):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape
