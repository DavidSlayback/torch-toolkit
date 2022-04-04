"""Modules that take in observations directly from a Gym or dm_env environment, with minimal preprocessing"""
from typing import Optional, Union, Sequence, Dict, overload, Any, Tuple

from functools import singledispatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import gym.spaces as spaces
import dm_env.specs as specs
SpaceOrSpec = Union[spaces.Space, specs.Array]

# Common keys
EMBED = 'embedding_size'

class OneHotLayer(nn.Module):
    """OneHot"""
    __constants__ = ['n']
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return F.one_hot(x, self.n)


class ReshapeLayer(nn.Module):
    """Reshape inputs of [T?, B?, ...] to [T?, B?, n]"""
    __constants__ = ['ndims_to_flatten']
    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self.ndims_to_flatten = len(shape)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        shape = (x.shape[:-self.ndims_to_flatten]) + (-1,)
        return torch.reshape(x, shape)


class ImageScaler(nn.Module):
    """Scale uint8 inputs by 255. No grad"""
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return x.float() / 255.


def build_intake_from_gym_env(envs: gym.Env, **kwargs) -> Tuple[nn.Module, Union[int, Sequence[int]]]:
    """Minimal intake for a gym observation. Return intake and its output size"""
    space = getattr(envs, 'single_observation_space', 'observation_space')  # Try to get single (vector env), otherwise full
    if isinstance(space, spaces.Discrete):
        return build_intake_from_discrete(space.n, **kwargs)
    elif isinstance(space, spaces.Box):
        is_float = np.issubdtype(space.dtype, np.inexact)
        if is_float: return build_intake_from_float_box(space)
        is_pixel = (np.issubdtype(space.dtype, np.unsignedinteger) and space.high.max() == 255)
        if is_pixel: return ImageScaler(), space.shape  # Assume that we need to scale if we're still seeing this space
        else: return build_intake_from_discrete_box(space)


def build_intake_from_discrete(n: int, **kwargs) -> Tuple[nn.Module, int]:
    """Embedding layer. Default to one-hot"""
    embed_dim = kwargs.pop(EMBED, 0)
    if not embed_dim: return OneHotLayer(n), n
    return nn.Embedding(n, embed_dim), embed_dim


def build_intake_from_float_box(space: spaces.Box) -> Tuple[nn.Module, int]:
    """Basic float box (e.g. MuJoCo). Flatten to vector of size [T?, B?, n] regardless"""
    return ReshapeLayer(space.shape), int(np.prod(space.shape))


def build_intake_from_discrete_box(space: spaces.Box, **kwargs) -> Tuple[nn.Module, int]:
    """Discrete Box (e.g., MiniGrid, FourRooms)"""
    embed_dim = kwargs.pop(EMBED, 0)
    n = space.high.max()  # Assume same high for all, could be smarter
    layers = [ReshapeLayer(space.shape)]  # Reshapes to [T?, B?, n]
    if not embed_dim: layers.append(OneHotLayer(n))
    else: layers.append(nn.Embedding(n, embed_dim))
    return nn.Sequential(*layers), embed_dim or n