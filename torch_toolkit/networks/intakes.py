__all__ = ['build_intake_from_gym_env']

"""Modules that take in observations directly from a Gym or dm_env environment, with minimal preprocessing"""
from typing import Union, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from gym import spaces, Env

from .misc import ImageScaler, OneHotLayer, FlattenLayer, FlattenDict

# Common keys
EMBED = 'embedding_size'


def get_cat_size(obs_space: spaces.Dict) -> int:
    return sum(v.shape[-1] if (len(v.shape) > 1) else 1 for v in obs_space.values())


def build_intake_from_gym_env(envs: Env, **kwargs) -> Tuple[nn.Module, Union[int, Sequence[int]]]:
    """Minimal intake for a gym observation. Return intake and its output size"""
    space = getattr(envs, 'single_observation_space', envs.observation_space)  # Try to get single (vector env), otherwise full
    if isinstance(space, spaces.Discrete):
        return build_intake_from_discrete(int(space.n), **kwargs)
    elif isinstance(space, spaces.Box):
        is_float = np.issubdtype(space.dtype, np.inexact)
        if is_float: return build_intake_from_float_box(space)
        is_pixel = (np.issubdtype(space.dtype, np.unsignedinteger) and space.high.max() == 255)
        if is_pixel: return ImageScaler(), space.shape  # Assume that we need to scale if we're still seeing this space
        else: return build_intake_from_discrete_box(space, **kwargs)
    elif isinstance(space, spaces.Dict):
        return build_intake_from_dict(space)
    else: raise ValueError('Single space must be box or discrete')


def build_intake_from_discrete(n: int, **kwargs) -> Tuple[nn.Module, int]:
    """Embedding layer. Default to one-hot"""
    embed_dim = kwargs.pop(EMBED, 0)
    if not embed_dim: return OneHotLayer(n), n
    return nn.Embedding(n, embed_dim), embed_dim


def build_intake_from_float_box(space: spaces.Box) -> Tuple[nn.Module, int]:
    """Basic float box (e.g. MuJoCo). Flatten to vector of size [T?, B?, n] regardless"""
    return FlattenLayer(space.shape), int(np.prod(space.shape))


def build_intake_from_discrete_box(space: spaces.Box, **kwargs) -> Tuple[nn.Module, int]:
    """Discrete Box (e.g., MiniGrid, FourRooms)"""
    embed_dim = kwargs.pop(EMBED, 0)
    n = int(space.high.max() + 1 - space.low.min())  # Assume same high for all, could be smarter
    numel = int(np.prod(space.shape))
    layers = [FlattenLayer(space.shape)]  # Reshapes to [T?, B?, n]  #  3x3 -> 9
    if not embed_dim: layers.append(OneHotLayer(n))
    else: layers.append(nn.Embedding(n, embed_dim))  # -> [9,8]
    layers.append(FlattenLayer((2, 2)))  # -> 9*8  Always 2 dimesions at end (1 for embeddings, one for overall size)
    return nn.Sequential(*layers), numel * (embed_dim or n)


def build_intake_from_dict(space: spaces.Dict, **kwargs) -> Tuple[nn.Module, int]:
    """Dict space (e.g., dm_control). Requires non-nested dict"""
    from ..utils import to_th, buffer_func
    itk = torch.jit.script(FlattenDict())
    example = itk(buffer_func(to_th(space.sample()), torch.unsqueeze, 0))  # Apply to example output to get size
    return itk, example.shape[-1]