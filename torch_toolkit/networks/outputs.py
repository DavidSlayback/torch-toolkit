"""Modules that output logits for a Gym or dm_env environment, with minimal postprocessing"""
__all__ = ['build_action_head_from_gym_env']

import collections
import math
from typing import Tuple, Optional, Union, Sequence, NamedTuple, Callable
from enum import IntEnum

import numpy as np
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from .init import layer_init, ORTHOGONAL_INIT_VALUES, ORTHOGONAL_INIT_VALUES_TORCH
from .misc import FlattenLayer, ReshapeLayer
import gym
from gym import spaces, Env
from ..utils import batched_index

Tensor = torch.Tensor


class ActionScaleType(IntEnum):
    none = 0  # No scaling
    post = 1  # Scaling after sample
    squash = 2  # Squash after the fact

def scale_raw_to_11(action: Tensor) -> Tensor: return torch.tanh(action)
def scale_11_to_01(action: Tensor) -> Tensor: return 0.5 * (action + 1.)
def scale_01_to_space(action: Tensor, range: Tensor, offset: Tensor) -> Tensor: return (action - offset) / range
def scale_raw_to_space(action: Tensor, range: Tensor, offset: Tensor) -> Tensor: return scale_01_to_space(scale_11_to_01(scale_raw_to_11(action)), range, offset)

# ActionHeadOutput = collections.namedtuple('ActionHeadOutput', ['action', 'scaled_action', 'logits'])
class ActionHeadOutput(NamedTuple):
    action: Tensor
    scaled_action: Tensor
    logits: Tensor


def build_action_head_from_gym_env(envs: Env,
                                   num_policies: int = 1,
                                   continuous_parameterization: str = 'beta',
                                   action_scaling: ActionScaleType = ActionScaleType.post
                                   ) -> Callable[[int], nn.Module]:
    """Return appropriate action head for an environment

    Args:
        envs: Gym Env/VectorEnv
        num_policies: How many sub-policies (i.e., options). Default to 1
        continuous_parameterization: String representation of continuous parameterization of action. 'beta', 'gaussian_ind', 'gaussian_dep'
        action_scaling: If post, scale action to env after sampling. If squash, do the SAC reparameterizing thing. Otherwise none
    Returns:
        Appropriate head module minus the input argument
    """
    space = getattr(envs, 'single_action_space', envs.action_space)
    if isinstance(space, spaces.Discrete):
        return partial(DiscreteHead, n=space.n, num_policies=num_policies)
    elif isinstance(space, spaces.MultiDiscrete):
        return partial(DiscreteHead, n=space.nvec, num_policies=num_policies)
    elif isinstance(space, spaces.Box):
        range = offset = None
        if action_scaling:
            range = space.high - space.low
            offset = space.low
        n = int(np.prod(space.shape))
        if continuous_parameterization == 'beta': return partial(BetaHead, n=n, num_policies=num_policies, range=range, offset=offset)
        elif continuous_parameterization == 'gaussian_ind': return partial(IndependentGaussianHead, n=n, num_policies=num_policies, range=range, offset=offset)
        elif continuous_parameterization == 'gaussian_dep': return partial(DependentGaussianHead, n=n, num_policies=num_policies, range=range, offset=offset)
        else: raise ValueError("Not a valid parameterization")
    else: raise ValueError('Not a supported action space')


import torch.distributions as dist

class DiscreteHead(nn.Module):
    """Categorical Head

    Args:
        in_size: Size of input from previous layer
        n: Size of output. If int, single discrete head. If Sequence, multi discrete
    """
    n_act: torch.jit.Final[int]
    num_policies: torch.jit.Final[int]
    def __init__(self, in_size: int, n: Union[int, Sequence[int]], num_policies: int = 1):
        super().__init__()
        self.n_act = n if isinstance(n, int) else int(np.max(n))
        self.num_policies = num_policies
        total_size = int(np.prod(n) * num_policies)
        linear = layer_init(nn.Linear(in_size, total_size), ORTHOGONAL_INIT_VALUES['pi'])
        if isinstance(n, Sequence) or num_policies > 1:
            reshape_size = (num_policies,) + tuple(n)
            self.pi = nn.Sequential(linear, ReshapeLayer(reshape_size))
        else: self.pi = linear

    def forward(self, x: Tensor) -> Tensor:
        """Get normalized logits (essentially log_softmax)"""
        logits = self.pi(x)
        return logits - logits.logsumexp(dim=-1, keepdim=True)

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Get action and logits"""
        logits = self.forward(x)
        if self.num_policies > 1:
            if idx is not None: logits = batched_index(idx, logits)
        action = torch.multinomial(self.probs(logits), 1).squeeze(-1)
        return action, F.one_hot(action, self.n_act), logits

    @torch.jit.export
    def probs(self, logits: Tensor):
        """Get all action probabilities"""
        return logits.exp()

    @torch.jit.export
    def log_prob(self, logits: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from normalized logits"""
        return logits.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    @torch.jit.export
    def entropy(self, logits: Tensor):
        """Get entropy from normalized logits"""
        return -(logits * logits.exp()).sum(-1)


class BetaHead(nn.Module):
    """Beta Head (continuous spaces)

    Args:
        in_size: Size of input from previous layer
        n: Size of output.
        num_policies: Number of sub policies
        range: If provided
    """
    n_act: torch.jit.Final[int]
    scale_action: torch.jit.Final[int]
    num_policies: torch.jit.Final[int]

    def __init__(self, in_size: int, n: int, num_policies: int = 1, range: Optional[np.ndarray] = None, offset: Optional[np.ndarray] = None):
        super().__init__()
        self.n_act = n
        self.num_policies = num_policies
        self.scale_action = (range is not None) and (offset is not None)
        total_size = int(n * num_policies * 2)
        linear = layer_init(nn.Linear(in_size, total_size), ORTHOGONAL_INIT_VALUES['pi'])
        if num_policies > 1: self.pi = nn.Sequential(linear, ReshapeLayer((num_policies, n, 2)))
        else: self.pi = nn.Sequential(linear, ReshapeLayer((n, 2)))
        if (range is None) or (offset is None): range = offset = 0
        self.range = nn.Parameter(torch.tensor(range), requires_grad=False)
        self.offset = nn.Parameter(torch.tensor(offset), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Get alpha and beta vectors (stacked on last dim)"""
        return F.softplus(self.pi(x)) + 1

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Get action and logits"""
        alphabeta = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        if self.num_policies > 1:
            if idx is not None: alphabeta = batched_index(idx, alphabeta)
        with torch.no_grad():
            action = torch._sample_dirichlet(alphabeta).select(-1, 0)
            if self.scale_action:
                scaled_action = scale_01_to_space(action, self.range, self.offset)
            else: scaled_action = action
        return action, scaled_action, alphabeta

    @torch.jit.export
    def probs(self, alphabeta: Tensor) -> Tensor:
        """Get all action probabilities TODO: Not implemented"""
        return alphabeta

    @torch.jit.export
    def log_prob(self, alphabeta: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from stacked alpha beta vector"""
        ht = torch.stack([action, 1. - action], -1)  # Heads or tails
        return ((torch.log(ht) * (alphabeta - 1.)).sum(-1) +
                torch.lgamma(alphabeta.sum(-1)) -
                torch.lgamma(alphabeta).sum(-1)).sum(-1)  # Sum over action dim

    @torch.jit.export
    def entropy(self, alphabeta: Tensor):
        """Get entropy from stacked alpha beta vector"""
        a0 = alphabeta.sum(-1)
        return (torch.lgamma(alphabeta).sum(-1) - torch.lgamma(a0) -
                (2 - a0) * torch.digamma(a0) -
                ((alphabeta - 1.0) * torch.digamma(alphabeta)).sum(-1)).sum(-1)  # Sum over action dim


class IndependentGaussianHead(nn.Module):
    """Gaussian Head with state-independent log_std

    Args:
        in_size: Size of input from previous layer
        n: Size of output.
        num_policies: Number of sub policies
        range: If provided
    """
    n_act: torch.jit.Final[int]
    scale_action: torch.jit.Final[int]
    squash: torch.jit.Final[bool] = False
    num_policies: torch.jit.Final[int]

    def __init__(self, in_size: int, n: int, num_policies: int = 1, range: Optional[np.ndarray] = None, offset: Optional[np.ndarray] = None, squash: bool = False):
        super().__init__()
        self.n_act = n
        self.scale_action = (range is not None) and (offset is not None)
        self.squash = squash
        self.num_policies = num_policies
        total_size = int(n * num_policies)
        linear = layer_init(nn.Linear(in_size, total_size), ORTHOGONAL_INIT_VALUES['pi'])
        if num_policies > 1:
            self.pi = nn.Sequential(linear, ReshapeLayer((num_policies, n)))
            self.log_std = nn.Parameter(torch.zeros(1, num_policies, n))
        else:
            self.pi = linear
            self.log_std = nn.Parameter(torch.zeros(1, n))
        if (range is None) or (offset is None): range = offset = 0
        self.range = nn.Parameter(torch.tensor(range), requires_grad=False)
        self.offset = nn.Parameter(torch.tensor(offset), requires_grad=False)
        # self._scale_action = scale_raw_to_space

    def forward(self, x: Tensor) -> Tensor:
        """Get mean and log std (stacked on last dim)"""
        mean = self.pi(x)
        return torch.stack([mean, self.log_std.expand_as(mean)], -1)

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Get action and logits"""
        meanlogstd = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        if self.num_policies > 1:
            if idx is not None: meanlogstd = batched_index(idx, meanlogstd)
        with torch.no_grad():
            action = torch.normal(meanlogstd[...,0], F.softplus(meanlogstd[...,1]))
            if self.scale_action: scaled_action = scale_raw_to_space(action, self.range, self.offset)
            else: scaled_action = action
        return action, scaled_action, meanlogstd

    @torch.jit.export
    def probs(self, meanlogstd: Tensor) -> Tensor:
        """Get all action probabilities TODO: Not implemented"""
        return meanlogstd

    @torch.jit.export
    def log_prob(self, meanlogstd: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from stacked mean log_std vector"""
        mean, log_std = meanlogstd.chunk(2, -1)
        var = log_std ** 2
        log_prob = (-((action - meanlogstd) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))).sum(-1)
        return log_prob

    @torch.jit.export
    def entropy(self, meanlogstd: Tensor):
        """Get entropy from stacked mean log_std vector"""
        mean, log_std = meanlogstd.chunk(2, -1)
        return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)


class DependentGaussianHead(nn.Module):
    """Gaussian Head with state-dependent log_std

    Args:
        in_size: Size of input from previous layer
        n: Size of output.
        num_policies: Number of sub policies
        range: If provided
    """
    n_act: torch.jit.Final[int]
    scale_action: torch.jit.Final[int]
    squash: torch.jit.Final[bool] = False
    num_policies: torch.jit.Final[int]

    def __init__(self, in_size: int, n: int, num_policies: int = 1, range: Optional[np.ndarray] = None,
                 offset: Optional[np.ndarray] = None, squash: bool = False):
        super().__init__()
        self.n_act = n
        self.scale_action = (range is not None) and (offset is not None)
        self.squash = squash
        self.num_policies = num_policies
        total_size = int(n * num_policies * 2)
        linear = layer_init(nn.Linear(in_size, total_size), ORTHOGONAL_INIT_VALUES['pi'])
        if num_policies > 1: self.pi = nn.Sequential(linear, ReshapeLayer((num_policies, n, 2)))
        else: self.pi = nn.Sequential(linear, ReshapeLayer((n, 2)))
        if (range is None) or (offset is None): range = offset = 0
        self.range = nn.Parameter(torch.tensor(range), requires_grad=False)
        self.offset = nn.Parameter(torch.tensor(offset), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Get mean and log std (stacked on last dim)"""
        return self.pi(x)

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Get action and logits"""
        meanlogstd = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        if self.num_policies > 1:
            if idx is not None: meanlogstd = batched_index(idx, meanlogstd)
        with torch.no_grad():
            action = torch.normal(meanlogstd[...,0], F.softplus(meanlogstd[...,1]))
            if self.scale_action: scaled_action = scale_raw_to_space(action, self.range, self.offset)
            else: scaled_action = action
        return action, scaled_action, meanlogstd

    @torch.jit.export
    def probs(self, meanlogstd: Tensor) -> Tensor:
        """Get all action probabilities TODO: Not implemented"""
        return meanlogstd

    @torch.jit.export
    def log_prob(self, meanlogstd: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from stacked mean log_std vector"""
        mean, log_std = meanlogstd.chunk(2, -1)
        var = log_std ** 2
        log_prob = (-((action - meanlogstd) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))).sum(-1)
        return log_prob

    @torch.jit.export
    def entropy(self, meanlogstd: Tensor):
        """Get entropy from stacked mean log_std vector"""
        mean, log_std = meanlogstd.chunk(2, -1)
        return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)


class BernoulliHead(nn.Module):
    """Bernoulli Head

    Args:
        in_size: Size of input from previous layer
        n: Size of output. Typically number of sub policies
    """
    n_act: torch.jit.Final[int]

    def __init__(self, in_size: int, n: int):
        super().__init__()
        self.n_act = n
        self.beta = layer_init(nn.Linear(in_size, n), ORTHOGONAL_INIT_VALUES['pi'])

    def forward(self, x: Tensor):
        return self.beta(x)

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Get action and logits (no sigmoid)"""
        logits = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        if idx is not None: logits = batched_index(idx, logits)
        with torch.no_grad(): action = torch.bernoulli(torch.sigmoid(logits))
        return action, action, logits

    @torch.jit.export
    def probs(self, logits: Tensor) -> Tensor:
        """Get all action probabilities"""
        return torch.sigmoid(logits)

    @torch.jit.export
    def log_prob(self, logits: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from non-sigmoid logits"""
        return -torch.binary_cross_entropy_with_logits(logits, action, reduction=0)

    @torch.jit.export
    def entropy(self, logits: Tensor):
        """Get entropy from non-sigmoid logits"""
        probs = torch.sigmoid(logits)
        return torch.binary_cross_entropy_with_logits(logits, probs, reduction=0)





