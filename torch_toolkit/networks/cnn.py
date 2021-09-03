__all__ = ['conv2d_output_shape', 'CNN', 'calculate_same_pad', 'ImageBOWEmbedding']

import math
from functools import partial
from typing import Tuple, Union, Optional, Sequence

import torch as th

Tensor = th.Tensor
import torch.nn as nn

Conv2dParam = Union[int, Tuple[int, int]]
Conv3dParam = Union[int, Tuple[int, int, int]]

from .init import layer_init, ORTHOGONAL_INIT_VALUES_TORCH
from .activation import maybe_inplace


def calculate_same_pad(input_h, input_w, strides, filter_h, filter_w):
    output_h = int(math.ceil(float(input_h) / float(strides[0])))
    output_w = int(math.ceil(float(input_w) / float(strides[1])))

    if input_h % strides[0] == 0:
        pad_along_height = max((filter_h - strides[0]), 0)
    else:
        pad_along_height = max(filter_h - (input_h % strides[0]), 0)
    if input_w % strides[1] == 0:
        pad_along_width = max((filter_w - strides[1]), 0)
    else:
        pad_along_width = max(filter_w - (input_w % strides[1]), 0)
    pad_top = pad_along_height // 2  # amount of zero padding on the top
    pad_bottom = pad_along_height - pad_top  # amount of zero padding on the bottom
    pad_left = pad_along_width // 2  # amount of zero padding on the left
    pad_right = pad_along_width - pad_left  # amount of zero padding on the right
    return pad_top + pad_bottom, pad_left + pad_right  # Pytorch doesn't take asymmetric


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class CNN(nn.Module):
    """Basic convolutional block

    Args:
        in_shape: c x h x w input shape
        channels: Number of out_channels for each convolutional laye
        kernel_sizes: Sizes of kernels for each convolutional layer
        strides: Strides for each convolutional layer
        paddings: Paddings for each convolutional layer. Defaults to none
        hidden_activation: nn.Module activation function. Defaults to relu
        flatten_end: Whether to flatten end. Defaults to false
        input_max: Maximum value in an input image. Defaults to 1. (assumes preprocessed)
        input_min: Minimum value in an input image. Defaults to 0. (assumes preprocessed)
    """

    def __init__(self,
                 input_shape: Sequence[int],
                 channels: Sequence[int],
                 kernel_sizes: Sequence[Conv2dParam],
                 strides: Sequence[Conv2dParam],
                 paddings: Optional[Sequence[Union[Conv2dParam, str]]] = None,
                 hidden_activation: nn.Module = nn.ReLU,
                 flatten_end: bool = False,
                 input_max: float = 1.,
                 input_min: float = 0.):
        super().__init__()
        c, h, w = input_shape
        l_init = partial(layer_init, std=ORTHOGONAL_INIT_VALUES_TORCH[hidden_activation])
        act = maybe_inplace(hidden_activation)
        in_channels = [c] + list(channels)[:-1]
        out_channels = list(channels)
        if paddings is None: paddings = [0 for _ in range(len(channels))]
        assert len(paddings) == len(in_channels) == len(out_channels) == len(kernel_sizes) == len(strides)
        layers = []
        for i, (ic, oc, k, s, p) in enumerate(zip(in_channels, out_channels, kernel_sizes, strides, paddings)):
            if p == 'same': p = calculate_same_pad(h, w, (s, s), k, k)  # Hack to fix same padding in PyTorch
            layers.extend([l_init(nn.Conv2d(ic, oc, k, s, p)), act()])
            h, w = conv2d_output_shape(h, w, k, s, p[0])
        if flatten_end: layers.append(nn.Flatten())
        self.conv = nn.Sequential(*layers)
        self._dim = self._conv_out_size(input_shape[1], input_shape[2], input_shape[0])
        if flatten_end: self._dim = int(math.prod(self._dim))
        self.do_scaling = not (input_max == 1 and input_min == 0)
        self.input_min = input_min;
        self.input_range = input_max - input_min

    def _scale_input(self, x: Tensor):
        return (x - self.input_min) / self.input_range

    def forward(self, x: Tensor):
        if self.do_scaling: x = self._scale_input(x)
        return self.conv(x)

    def _conv_out_size(self, h, w, c=None):
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                                           child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return c, h, w


class ImageBOWEmbedding(nn.Module):
    """Image Bagof-words embedding from BabyAI https://github.com/mila-iqia/babyai/blob/863f3529371ba45ef0148a48b48f5ae6e61e06cc/babyai/model.py#L48

    Useful for converting MiniGrid observations.
    Args:
        max_value: Maximum value in observation space
        embedding_dim: Dimension of embedding
        in_channel: Number of in channels. Assumed 3 (MiniGrid)
    """
    def __init__(self, max_value, embedding_dim, in_channel: int = 3):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(in_channel * max_value, embedding_dim)
        self.register_buffer('offsets', th.tensor([i * max_value for i in range(in_channel)]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0))


    def forward(self, x: Tensor):
        x.add_(self.offsets.expand_as(x))
        return self.embedding(x.long()).sum(1).permute(0, 3, 1, 2)


AGAC_CNN = partial(CNN, channels=(32, 32, 32), kernel_sizes=(3, 3, 3), strides=(2, 2, 2), paddings=('same',) * 3,
                   hidden_activation=nn.ELU)
