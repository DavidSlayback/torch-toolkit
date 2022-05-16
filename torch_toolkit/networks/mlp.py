__all__ = ['MLP']

from functools import partial
from typing import Sequence

import torch.nn as nn
from torch.nn import Linear

from .activation_util import maybe_inplace
from .init import layer_init, ORTHOGONAL_INIT_VALUES_TORCH
from ..typing import Tensor


class MLP(nn.Module):
    """Basic MLP.

    Args:
        in_size: Input size
        hidden_sizes: List of hidden sizes
        hidden_activation: nn Module activation function
        layer_norm_input: Apply layer normalization to input
    """
    def __init__(self,
                 in_size: int,
                 hidden_sizes: Sequence[int],
                 hidden_activation: nn.Module = nn.ReLU,
                 layer_norm_input: bool = False):
        super().__init__()
        l_init = partial(layer_init, std=ORTHOGONAL_INIT_VALUES_TORCH[hidden_activation])
        hidden_activation = maybe_inplace(hidden_activation)
        h = [in_size] + list(hidden_sizes)
        layers = []
        for i in range(len(h)-1): layers.extend([l_init(Linear(h[i], h[i+1])), hidden_activation()])
        if layer_norm_input: layers = [nn.LayerNorm(in_size)] + layers
        self.core = nn.Sequential(*layers)
        self._dim = h[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.core(x)
