"""Modules that output logits for a Gym or dm_env environment, with minimal postprocessing"""
__all__ = ['build_action_head_from_gym_env']

from typing import Tuple, Optional, Union, Sequence

import numpy as np
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from .init import layer_init, ORTHOGONAL_INIT_VALUES, ORTHOGONAL_INIT_VALUES_TORCH
from .misc import FlattenLayer, ReshapeLayer
import gym
from gym import spaces, Env

Tensor = torch.Tensor

def build_action_head_from_gym_env(envs: Env, num_policies: int = 1, continuous_parameterization: str = 'beta'):
    """Return appropriate action head for an environment

    Args:
        envs: Gym Env/VectorEnv
        num_policies: How many sub-policies (i.e., options). Default to 1
        continuous_parameterization: String representation of continuous parameterization of action
    Returns:
        Appropriate head module minus the input argument
    """
    space = getattr(envs, 'single_action_space', envs.action_space)
    if isinstance(space, spaces.Discrete):
        return partial(DiscreteHead, n=space.n, num_policies=num_policies)
    elif isinstance(space, spaces.MultiDiscrete):
        return partial(DiscreteHead, n=space.nvec, num_policies=num_policies)
    elif isinstance(space, spaces.Box):
        if continuous_parameterization == 'beta': return partial(BetaHead, n=int(np.prod(space.shape)), num_policies=num_policies)
        else: raise ValueError


import torch.distributions as dist

dist.Categorical
dist.Beta


class DiscreteHead(nn.Module):
    """Categorical Head

    Args:
        in_size: Size of input from previous layer
        n: Size of output. If int, single discrete head. If Sequence, multi discrete
    """
    def __init__(self, in_size: int, n: Union[int, Sequence[int]], num_policies: int = 1):
        super().__init__()
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
    @torch.no_grad()
    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get action and logits"""
        logits = self.forward(x)
        return torch.multinomial(self.probs(logits), 1).squeeze(-1), logits

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
    """
    def __init__(self, in_size: int, n: int, num_policies: int = 1):
        super().__init__()
        total_size = int(n * num_policies * 2)
        linear = layer_init(nn.Linear(in_size, total_size), ORTHOGONAL_INIT_VALUES['pi'])
        if num_policies > 1: self.pi = nn.Sequential(linear, ReshapeLayer((num_policies, n, 2)))
        else: self.pi = nn.Sequential(linear, ReshapeLayer((n, 2)))

    def forward(self, x: Tensor) -> Tensor:
        """Get alpha and beta vectors (stacked on last dim)"""
        return F.softplus(self.pi(x)) + 1

    @torch.jit.export
    @torch.no_grad()
    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get action and logits"""
        alphabeta = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        return torch._sample_dirichlet(alphabeta).select(-1, 0), alphabeta

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







