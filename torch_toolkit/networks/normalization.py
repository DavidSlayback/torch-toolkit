__all__ = ['ObservationNormalizationModule', 'RMSNorm']

import numbers
from typing import Tuple, Union, List, Optional, Final
import torch
import torch.nn as nn
from ..typing import Tensor

_shape_t = Union[int, List[int], torch.Size]


class RMSNorm(nn.Module):
    """Root-mean-squared normalization layer

    An alternative to LayerNorm proposed here: https://openreview.net/pdf?id=SygkZ3MTJE
    Instead of recentering and rescaling inputs, just rescale. Faster with comparalbe or better performance

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        p: For pRMSNorm. First p% of inputs are used to compute statistics instead of all inputs
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', 'p', 'd']
    normalized_shape: Tuple[int, ...]
    eps: float
    p: float
    d: int
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, p: float = 1., eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.d = self.normalized_shape[0]
        self.p = p
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine: self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else: self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine: nn.init.ones_(self.weight)

    def forward(self, x: Tensor):
        """Scale by RMS norm or pRMSNorm

        a_i = a_i / RMS(a) g_i
        RMS(a) = sqrt((1/n) sum_{i=1:n}a_i^2)
        pRMS(a) = sqrt((1/k) sum_{i=1:k}a_i^2), k=ceil(n*p)
        """
        if self.p >= 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


class ObservationNormalizationModule(nn.Module):
    """Standalone PyTorch module to normalize observations (without gradient)

    Args:
        n_obs: Number of observations from environment. Assumes vector
        normalize_obs: If True,
    """
    normalize_obs: Final[bool] = True
    def __init__(self, n_obs: int, normalize_obs: bool = True):
        super().__init__()
        self.n_obs = n_obs
        self.normalize_obs = normalize_obs
        self.num_steps = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.running_mean = nn.Parameter(torch.zeros(n_obs), requires_grad=False)
        self.running_variance = nn.Parameter(torch.zeros(n_obs), requires_grad=False)

    @torch.no_grad()
    def forward(self, x: Tensor, running_update: bool = False) -> Tensor:
        """Normalize observation. Optionally, update statistics before"""
        if self.normalize_obs:
            if running_update: self.update_normalization(x)
            return self.normalize(x)
        else: return x

    @torch.jit.export
    @torch.no_grad()
    def update_normalization(self, x: Tensor) -> None:
        """Update mean, variance, and steps"""
        if self.normalize_obs:
            xv = x.view(-1, self.n_obs)
            self.num_steps.add_(xv.shape[0])
            input_to_old_mean = xv - self.running_mean
            mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=0)
            self.running_mean.add_(mean_diff)
            input_to_new_mean = xv - self.running_mean
            var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=0)
            self.running_variance.add_(var_diff)

    @torch.jit.export
    def normalize(self, x: Tensor) -> Tensor:
        """Normalize input tensor with current statistics"""
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((x - self.running_mean) / variance.sqrt()).clip(-5, 5)
