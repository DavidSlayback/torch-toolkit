__all__ = ['Tanh', 'Sigmoid', 'LogSigmoid']

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..typing import Tensor


# In practice, I've found that in-place activations give a marked speedup
class Tanh(nn.Module):
    """Same as nn.Tanh, with option to perform inplace"""
    __constants__ = ['inplace']
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.fn = th.tanh_ if inplace else th.tanh

    def forward(self, input: Tensor) -> Tensor:
        return self.fn(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Sigmoid(nn.Module):
    """Same as nn.Sigmoid, with option to perform inplace"""
    __constants__ = ['inplace']
    inplace: bool
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.fn = th.sigmoid_ if inplace else th.sigmoid

    def forward(self, input: Tensor) -> Tensor:
        return self.fn(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class LogSigmoid(nn.Module):
    __constants__ = ['inplace']
    inplace: bool
    """Same as nn.LogSigmoid, with option to perform inplace. May not be benefiting from stability, no logsigmoid_"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid_().log_() if self.inplace else F.logsigmoid(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

