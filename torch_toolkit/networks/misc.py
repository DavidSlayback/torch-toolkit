__all__ = ['OneHotLayer', 'FlattenLayer', 'ImageScaler', 'ReshapeLayer']

from typing import Sequence, Iterable, Dict, Tuple, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..typing import Tensor


class OneHotLayer(nn.Module):
    """OneHot"""
    n: Final[int]
    def __init__(self, n: int):
        super().__init__()
        self.n = int(n)

    @torch.no_grad()
    def forward(self, x: Tensor):
        return F.one_hot(x, self.n)


class FlattenLayer(nn.Module):
    """Reshape inputs of [T?, B?, ...] to [T?, B?, n]"""
    ndims_to_flatten: Final[int]
    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self.ndims_to_flatten = len(shape)

    def forward(self, x: Tensor):
        shape = (x.shape[:-self.ndims_to_flatten]) + (-1,)
        return torch.reshape(x, shape)

class ReshapeLayer(nn.Module):
    """Reshape inputs of [T?, B?, in_shape] to [T?, B?, out_shape]"""
    shape: Final[Tuple[int]]
    def __init__(self, shape: Sequence[int]):
        super(ReshapeLayer, self).__init__()
        self.shape = tuple(shape)

    def forward(self, x: Tensor):
        """Get leading dims, reshape"""
        return torch.reshape(x, x.shape[:-1] + self.shape)



class FlattenDict(nn.Module):
    """Reshape inputs from dictionary to flat vector

    NOTE: Requires first value in dict to be [T?, B?, n] instead of [T?, B?]
    """
    @torch.no_grad()
    def forward(self, x: Dict[str, Tensor]):
        tensors = [v.float() for v in x.values()]  # Get tensors as float
        shape = tensors[0].shape[:-1] + (-1,)  # TODO: Don't require first key to be non-scalar
        return torch.cat([torch.reshape(v, shape) for v in tensors], -1)



class ImageScaler(nn.Module):
    """Scale uint8 inputs by 255. No grad"""
    @torch.no_grad()
    def forward(self, x: Tensor):
        return x.float() / 255.