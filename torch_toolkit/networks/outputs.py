"""Modules that output logits for a Gym or dm_env environment, with minimal postprocessing"""
__all__ = ['build_action_head_from_gym_env',
           'build_separate_ff_actor_critic', 'build_separate_ff_option_actor',
           'ActorCritic_Unshared', 'OptionCritic_Unshared', 'FFActor', 'GRUActor', 'FFFullNet', 'GRUFullNet',
           'build_mlp', 'InterestHead', 'BernoulliHead', 'build_separate_ff_interest']

import math
from enum import IntEnum
from functools import partial
from typing import Tuple, Optional, Union, Sequence, NamedTuple, Callable, Dict, Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces, Env

from .init import layer_init, ORTHOGONAL_INIT_VALUES, ORTHOGONAL_INIT_VALUES_TORCH
from .misc import ReshapeLayer
from ..utils import batched_index

Tensor = torch.Tensor

Action = Tensor  # Direct action sample
ScaledAction = Tensor  # "Scaled action" appropriate for input to environment or network
Logits = Tensor  # Raw logits before sampling

class ActorOutput(NamedTuple):
    a: Tensor
    s_a: Tensor
    logits: Tensor


class ActorHead(nn.Module):
    """General actor head from which all actor heads inherit"""

    def forward(self, x: Tensor) -> Tensor:
        return self.pi(x)


    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None, all_logits: bool = False) -> ActorOutput:
        """Sample actions given input

        Args:
            x: Input from some previous network body
            idx: Indices to use to subselect logits (e.g., options in option-critic)
                If there's only one policy, then these ar ignored
            all_logits: If true, return logits for all sub policies
        """
        raise NotImplementedError

    @torch.jit.export
    def probs(self, logits: Logits) -> Tensor:
        """Policy-specific probabilities given raw logits"""
        raise NotImplementedError

    @torch.jit.export
    def log_prob(self, logits: Logits, action: Action) -> Tensor:
        """Policy-specific log probabilities of action given raw logits

        Args:
            logits: [T?,B,n_polcies?,n_A]
            action: [T?, B, n_A?]
        """
        raise NotImplementedError

    @torch.jit.export
    def entropy(self, logits: Logits) -> Tensor:
        """Policy-specific entropy of distribution given raw logits"""
        raise NotImplementedError

class ActionScaleType(IntEnum):
    none = 0  # No scaling
    post = 1  # Scaling after sample
    squash = 2  # Squash after the fact

# String keys
PI_KEY = 'actor'
CRITIC_KEY = 'critic'
OPTION_ACTOR_KEY = 'actor_w'
TERMINATION_KEY = 'termination'

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
                                   ) -> Callable[[int], ActorHead]:
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


class DiscreteHead(ActorHead):
    """Categorical Head

    Args:
        in_size: Size of input from previous layer
        n: Size of output. If int, single discrete head. If Sequence, multi discrete
    """
    n_act: Final[int]
    num_policies: Final[int]

    def __init__(self, in_size: int, n: Union[int, Sequence[int]], num_policies: int = 1):
        super().__init__()
        self.n_act = n if isinstance(n, int) else int(np.max(n))
        self.num_policies = num_policies
        total_size = int(np.prod(n) * num_policies)
        linear = layer_init(nn.Linear(in_size, total_size), ORTHOGONAL_INIT_VALUES['pi'])
        if isinstance(n, Sequence) or num_policies > 1:
            reshape_size = (num_policies,) + (tuple(n) if isinstance(n, Sequence) else (n,))
            self.pi = nn.Sequential(linear, ReshapeLayer(reshape_size))
        else: self.pi = linear

    def forward(self, x: Tensor) -> Tensor:
        """Get normalized logits (essentially log_softmax)"""
        logits = self.pi(x)
        return logits - logits.logsumexp(dim=-1, keepdim=True)

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None, all_logits: bool = False) -> ActorOutput:
        """Get action, one hot action, and logits"""
        logits = self.forward(x)
        s_logits = logits
        if self.num_policies > 1:
            if idx is not None: s_logits = batched_index(idx, logits)
        action = torch.multinomial(self.probs(s_logits), 1).squeeze(-1)
        return ActorOutput(action, action, logits if all_logits else s_logits)

    @torch.jit.export
    def probs(self, logits: Tensor):
        """Get all action probabilities"""
        return logits.exp()

    @torch.jit.export
    def log_prob(self, logits: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from normalized logits

        If we get logits for all policies, return logprobs for actions under all policies
        Otherwise just return logprobs for the logits we get
        Args:
            logits: [T?, B, n_policy?, n_act] log probabilities
            action: [T?, B] actions.
        Returns:
            [T?, B, n_policy?] log probabilities of sampled action
        """
        if self.num_policies > 1: return torch.take_along_dim(logits, action.reshape(logits.shape[:-2] + (1,1)), -1).squeeze(-1)
        else: return logits.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    @torch.jit.export
    def entropy(self, logits: Tensor):
        """Get entropy from normalized logits"""
        return -(logits * logits.exp()).sum(-1)


class BetaHead(ActorHead):
    """Beta Head (continuous spaces)

    Args:
        in_size: Size of input from previous layer
        n: Size of output.
        num_policies: Number of sub policies
        range: If provided
    """
    n_act: Final[int]
    scale_action: Final[int]
    num_policies: Final[int]

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
    def sample(self, x: Tensor, idx: Optional[Tensor] = None, all_logits: bool = False) -> ActorOutput:
        """Get action and logits"""
        alphabeta = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        s_alphabeta = alphabeta
        if self.num_policies > 1:
            if idx is not None: s_alphabeta = batched_index(idx, alphabeta)
        with torch.no_grad():
            action = torch._sample_dirichlet(s_alphabeta).select(-1, 0)
            if self.scale_action:
                scaled_action = scale_01_to_space(action, self.range, self.offset)
            else: scaled_action = action
        return ActorOutput(action, scaled_action, alphabeta if all_logits else s_alphabeta)

    @torch.jit.export
    def probs(self, alphabeta: Tensor) -> Tensor:
        """Get all action probabilities TODO: Not implemented"""
        return alphabeta

    @torch.jit.export
    def log_prob(self, alphabeta: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from stacked alpha beta vector

        If we get alphabeta for all policies, return logprobs for actions under all policies
        Otherwise just return logprobs for the logits we get
        Args:
            alphabeta: [T?, B, n_policy?, n_act, 2] alphabeta vector
            action: [T?, B, n_act] actions.
        """
        if self.num_policies > 1 and (alphabeta.ndim - action.ndim == 2): action = action.unsqueeze(-2)  # Get logprob for each subpolicy
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


class IndependentGaussianHead(ActorHead):
    """Gaussian Head with state-independent log_std

    Args:
        in_size: Size of input from previous layer
        n: Size of output.
        num_policies: Number of sub policies
        range: If provided
    """
    n_act: Final[int]
    scale_action: Final[int]
    squash: Final[bool] = False
    num_policies: Final[int]

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
    def sample(self, x: Tensor, idx: Optional[Tensor] = None, all_logits: bool = False) -> ActorOutput:
        """Get action and logits"""
        meanlogstd = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        s_meanlogstd = meanlogstd
        if self.num_policies > 1:
            if idx is not None: s_meanlogstd = batched_index(idx, meanlogstd)
        with torch.no_grad():
            action = torch.normal(s_meanlogstd[...,0], F.softplus(s_meanlogstd[...,1]))
            if self.scale_action: scaled_action = scale_raw_to_space(action, self.range, self.offset)
            else: scaled_action = action
        return ActorOutput(action, scaled_action, meanlogstd if all_logits else s_meanlogstd)

    @torch.jit.export
    def probs(self, meanlogstd: Tensor) -> Tensor:
        """Get all action probabilities TODO: Not implemented"""
        return meanlogstd

    @torch.jit.export
    def log_prob(self, meanlogstd: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from stacked mean log_std vector"""
        if self.num_policies > 1 and (meanlogstd.ndim - action.ndim == 2): action = action.unsqueeze(-2)  # Get logprob for each subpolicy
        mean, log_std = meanlogstd.chunk(2, -1)
        mean = mean.squeeze(-1)
        log_std = log_std.squeeze(-1)
        var = log_std ** 2
        log_prob = (-((action - mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))).sum(-1)
        return log_prob

    @torch.jit.export
    def entropy(self, meanlogstd: Tensor):
        """Get entropy from stacked mean log_std vector"""
        log_std = meanlogstd[..., -1]
        return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)


class DependentGaussianHead(ActorHead):
    """Gaussian Head with state-dependent log_std

    Args:
        in_size: Size of input from previous layer
        n: Size of output.
        num_policies: Number of sub policies
        range: If provided
    """
    n_act: Final[int]
    scale_action: Final[int]
    squash: Final[bool] = False
    num_policies: Final[int]

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

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None, all_logits: bool = False) -> ActorOutput:
        """Get action and logits"""
        meanlogstd = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        s_meanlogstd = meanlogstd
        if self.num_policies > 1:
            if idx is not None: s_meanlogstd = batched_index(idx, meanlogstd)
        with torch.no_grad():
            action = torch.normal(s_meanlogstd[..., 0], F.softplus(s_meanlogstd[...,1]))
            if self.scale_action: scaled_action = scale_raw_to_space(action, self.range, self.offset)
            else: scaled_action = action
        return ActorOutput(action, scaled_action, meanlogstd if all_logits else s_meanlogstd)

    @torch.jit.export
    def probs(self, meanlogstd: Tensor) -> Tensor:
        """Get all action probabilities TODO: Not implemented"""
        return meanlogstd

    @torch.jit.export
    def log_prob(self, meanlogstd: Tensor, action: Tensor) -> Tensor:
        """Get log probability of action from stacked mean log_std vector"""
        if self.num_policies > 1 and (meanlogstd.ndim - action.ndim == 2): action = action.unsqueeze(-2)  # Get logprob for each subpolicy
        mean, log_std = meanlogstd.chunk(2, -1)
        mean = mean.squeeze(-1)
        log_std = log_std.squeeze(-1)
        var = log_std ** 2
        log_prob = (-((action - mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))).sum(-1)
        return log_prob

    @torch.jit.export
    def entropy(self, meanlogstd: Tensor):
        """Get entropy from stacked mean log_std vector"""
        log_std = meanlogstd[..., -1]
        return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)


class BernoulliHead(ActorHead):
    """Bernoulli Head

    Args:
        in_size: Size of input from previous layer
        n: Size of output. Typically number of sub policies
    """
    n_act: Final[int]

    def __init__(self, in_size: int, n: int):
        super().__init__()
        self.n_act = n
        # self.pi = layer_init(nn.Linear(in_size, n), ORTHOGONAL_INIT_VALUES['pi'])
        self.pi = layer_init(nn.Linear(in_size, n), ORTHOGONAL_INIT_VALUES['sigmoid'])

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None, all_logits: bool = True) -> ActorOutput:
        """Get action and logits (no sigmoid)"""
        logits = self.forward(x)  # [T?,B?,n_policies?, n, 2]
        s_logits = logits
        if idx is not None: s_logits = batched_index(idx, logits)
        with torch.no_grad(): action = torch.bernoulli(torch.sigmoid(s_logits))
        return ActorOutput(action, action, logits if all_logits else s_logits)

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


class CriticHead(nn.Module):
    """Value head"""
    def __init__(self, in_size: int, n: int = 1):
        super().__init__()
        self.critic = layer_init(nn.Linear(in_size, n), 1.)

    def forward(self, x: Tensor) -> Tensor: return self.critic(x).squeeze(-1)


class InterestHead(nn.Module):
    """Sigmoid interest function"""
    def __init__(self, in_size: int, n: int):
        super().__init__()
        # self.interest = layer_init(in_size, n, ORTHOGONAL_INIT_VALUES['pi'])
        self.interest = layer_init(nn.Linear(in_size, n), ORTHOGONAL_INIT_VALUES['pi'])

    def forward(self, x: Tensor) -> Tensor: return torch.sigmoid(self.interest(x))

from .activations import Tanh
from .activation_util import maybe_inplace


class FFFullNet(nn.Module):
    """Critic class with FF body"""
    def __init__(self, body: nn.Module, head: nn.Module):
        super().__init__()
        self.body = body
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


class GRUFullNet(FFFullNet):
    """Critic class with gru body"""
    def forward(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        state = self.body(x, state)
        return self.head(self.body(state)), state


class FFActor(nn.Module):
    """Actor class with FF body"""
    def __init__(self, body: nn.Module, head: ActorHead):
        super().__init__()
        self.body = body
        self.head = head

    def forward(self, x: Tensor):
        return self.head(self.body(x))

    @torch.jit.export
    def sample(self, x: Tensor, idx: Optional[Tensor] = None, all_logits: bool = False) -> ActorOutput:
        return self.head.sample(self.body(x), idx, all_logits)

    @torch.jit.export
    def probs(self, logits: Tensor) -> Tensor: return self.head.probs(logits)

    @torch.jit.export
    def log_probs(self, logits: Tensor, action: Tensor) -> Tensor: return self.head.log_prob(logits, action)

    @torch.jit.export
    def entropy(self, logits: Tensor) -> Tensor: return self.head.entropy(logits)


class GRUActor(FFActor):
    """Actor class with GRU body"""
    def forward(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        state = self.body(x, state)
        return self.head(self.body(state)), state

    @torch.jit.export
    def sample(self, x: Tensor, state: Tensor, idx: Optional[Tensor] = None, all_logits: bool = False) -> Tuple[ActorOutput, Tensor]:
        state = self.body(x, state)
        return self.head.sample(self.body(state), idx, all_logits), state


def build_mlp(in_size: int, hidden_sizes: Sequence[int], hidden_activation: Callable[[], nn.Module] = Tanh) -> nn.Module:
    """Build FF MLP"""
    act_module = maybe_inplace(hidden_activation)
    act_std = ORTHOGONAL_INIT_VALUES_TORCH[hidden_activation]
    sizes = [in_size,] + list(hidden_sizes)
    layers = []
    for s0, s1 in zip(sizes[:-1], sizes[1:]):
        layers.extend([layer_init(nn.Linear(s0, s1), act_std), act_module()])
    body = nn.Sequential(*layers)
    return body


def build_separate_ff_actor(envs: Env, in_size: int, hidden_sizes: Sequence[int], hidden_activation: Callable[[], nn.Module] = Tanh, num_policies: int = 1, continuous_parameterization='beta') -> FFActor:
    """Build feedforward actor module"""
    head = build_action_head_from_gym_env(envs, num_policies=num_policies, continuous_parameterization=continuous_parameterization)(hidden_sizes[-1])
    return FFActor(build_mlp(in_size, hidden_sizes, hidden_activation), head)


def build_separate_ff_option_actor(in_size: int, hidden_sizes: Sequence[int], hidden_activation: Callable[[], nn.Module] = Tanh, num_policies: int = 1) -> FFActor:
    """Discrete option policy head"""
    head = DiscreteHead(hidden_sizes[-1], num_policies)
    return FFActor(build_mlp(in_size, hidden_sizes, hidden_activation), head)


def build_separate_ff_termination(in_size: int, hidden_sizes: Sequence[int], hidden_activation: Callable[[], nn.Module] = Tanh, num_policies: int = 1) -> FFActor:
    """Bernoulli"""
    head = BernoulliHead(hidden_sizes[-1], num_policies)
    return FFActor(build_mlp(in_size, hidden_sizes, hidden_activation), head)


def build_separate_ff_critic(in_size: int, hidden_sizes: Sequence[int], hidden_activation: Callable[[], nn.Module] = Tanh, num_policies: int = 1) -> FFFullNet:
    """Critic (V or Q)"""
    head = layer_init(nn.Linear(hidden_sizes[-1], num_policies), 1.)
    return FFFullNet(build_mlp(in_size, hidden_sizes, hidden_activation), head)


def build_separate_ff_interest(in_size: int, hidden_sizes: Sequence[int], hidden_activation: Callable[[], nn.Module] = Tanh, num_policies: int = 1) -> FFFullNet:
    """Interest function (like termination, but no sampling"""
    head = InterestHead(hidden_sizes[-1], num_policies)
    return FFFullNet(build_mlp(in_size, hidden_sizes, hidden_activation), head)


class ActorCritic_Unshared(nn.Module):
    """Separate actor and critic networks. Could be entire network, could just be unshared component"""
    def __init__(self, actor: FFActor, critic: FFFullNet):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.actor(x), self.critic(x)

    @torch.jit.export
    def sample(self, x: Tensor) -> Dict[str, Tensor]:
        a, s_a, logits = self.actor.sample(x)
        lp = self.actor.log_probs(logits, a)
        return {'a': a, 's_a': s_a, 'lp': lp}

    @torch.jit.export
    def unroll(self, x: Tensor, action: Tensor) -> Dict[str, Tensor]:
        # T, B = x.shape[:2]
        # x = x.view(T*B, -1)
        logits = self.actor.forward(x)
        lp = self.actor.log_probs(logits, action)
        ent = self.actor.entropy(logits)
        v = self.critic(x)
        return {'lp': lp, 'ent': ent, 'v': v}
        # return {'lp': lp.view(T, B), 'ent': ent.view(T, B), 'v': v.view(T, B)}


class OptionCritic_Unshared(nn.Module):
    """Separate actor, critic, option actor, termination networks. Could be entire network, could just be unshared component

    Critic and option actor inputs may be different (e.g., FA2OC)
    """
    separate_option_input: Final[bool] = False
    separate_critic_input: Final[bool] = False

    def __init__(self, actor: FFActor, critic: FFFullNet, termination: FFActor, actor_w: FFActor,
                 separate_option_input: bool = False,
                 separate_critic_input: bool = False):
        super().__init__()
        self.actor = actor
        self.actor_w = actor_w
        self.termination = termination
        self.critic = critic
        self.separate_critic_input = separate_critic_input
        self.separate_option_input = separate_option_input

    def forward(self, x: Tensor, x_w: Optional[Tensor] = None, x_q: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pi, beta, q, pi_w = self.actor(x), self.termination(x), self._critic(x, x_q), self._option_actor(x, x_w)
        return pi, q, beta, pi_w

    def _option_actor(self, x: Tensor, x_w: Optional[Tensor] = None):
        if self.separate_option_input and (x_w is not None): return self.actor_w(x_w)
        else: return self.actor_w(x)

    def _critic(self, x: Tensor, x_q: Optional[Tensor] = None):
        if self.separate_critic_input and (x_q is not None): return self.critic(x_q)
        else: return self.critic(x)

    def _sample_option_actor(self, x: Tensor, x_w: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        if self.separate_option_input and (x_w is not None): return self.actor_w.sample(x_w)
        else: return self.actor_w.sample(x)

    def _sample_actor(self, x: Tensor, idx: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.actor.sample(x, idx)

    @torch.jit.export
    def sample(self, x: Tensor, prev_option: Tensor, should_reset: Tensor, x_w: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Sample termination, new option, action from option"""
        termination, beta_logits = self.termination.sample(x=x, idx=prev_option)[0:3:2]  # Sample termination for previous option
        new_option, option_logits = self._sample_option_actor(x, x_w)[0:3:2]
        option = torch.where((termination + should_reset) > 0, new_option, prev_option)  # Sample option where terminal
        action, scaled_action, action_logits = self._sample_actor(x, option)
        lp = self.actor.log_probs(action_logits, action)
        lp_w = self.actor_w.log_probs(option_logits, option)
        return {'t': termination, 'b': beta_logits,
                'w': option, 'lp_w': lp_w,
                'a': action, 's_a': scaled_action, 'lp': lp}

    @torch.jit.export
    def sample_without_action(self, x: Tensor, prev_option: Tensor, should_reset: Tensor, x_w: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Sample termination and new option, don't sample action (wait for new x input as in FA2OC"""
        termination, beta_logits = self.termination.sample(x=x, idx=prev_option)[0:3:2]  # Sample termination for previous option
        new_option, option_logits = self._sample_option_actor(x, x_w)[0:3:2]
        option = torch.where((termination + should_reset) > 0, new_option, prev_option)  # Sample option where terminal
        lp_w = self.actor_w.log_probs(option_logits, new_option)
        return {'t': termination, 'b': beta_logits, 'w': option, 'lp_w': lp_w}

    @torch.jit.export
    def sample_action_from_option(self, x: Tensor, option: Tensor) -> Dict[str, Tensor]:
        action, scaled_action, action_logits = self._sample_actor(x, option)
        lp = self.actor.log_probs(action_logits, action)
        return {'a': action, 's_a': scaled_action, 'lp': lp}

    @torch.jit.export
    def unroll(self, x: Tensor, prev_option: Tensor, option: Tensor, action: Tensor,
               x_w: Optional[Tensor] = None, x_q: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Get all necessary quantities of unroll

        Options is T+1
        First T are prev option, T+1 are option

        x is T+1
        action is T+1
        """
        # T, B = x.shape[:2]
        # x = x.view(T*B, -1)
        # p_w = prev_option.view(T*B)
        # w = option.view(T*B)
        # Termination component
        beta_logits = self.termination.forward(x[:-1])
        beta_probs = self.termination.probs(beta_logits)  # All termination probabilities
        beta_probs_w_tm1 = batched_index(prev_option, beta_probs)  # Termination probabilities of previous option
        # Option component
        option_logits = self._option_actor(x, x_w)
        option_probs = self.actor_w.probs(option_logits)  # All option probabilities
        lp_w = self.actor_w.log_probs(option_logits, option)  # log prob of chosen options
        ent_w = self.actor_w.entropy(option_logits)  # Entropy of option distribution
        # Actor component
        actor_logits = batched_index(option, self.actor.forward(x))  # Only action logits for chosen option
        lp = self.actor.log_probs(actor_logits, action)
        ent = self.actor.probs(actor_logits)
        # Critic component
        q = self._critic(x, x_q)  # Value of all options (to calculate V)
        q_w_t = batched_index(option, q)  # Value of chosen option (for critic loss)
        q_w_tm1 = batched_index(prev_option, q)  # Value of previous option (for option and termination policy losses)
        return {'b': beta_probs, 'b_tm1': beta_probs_w_tm1,
                'lp_w': lp_w, 'ent_w': ent_w, 'p_w': option_probs,
                'lp': lp, 'ent': ent,
                'q': q, 'q_w_t': q_w_t, 'q_w_tm1': q_w_tm1}


def build_separate_ff_actor_critic(envs: Env,
                                   in_size: int,
                                   hidden_sizes: Union[Sequence[int], Dict[str, Sequence[int]]],
                                   hidden_activation: Callable[[], nn.Module] = Tanh,
                                   continuous_parameterization='beta') -> Tuple[FFActor, FFFullNet]:
    """Actor and critic networks"""
    if isinstance(hidden_sizes, Sequence): hidden_sizes = {PI_KEY: hidden_sizes, CRITIC_KEY: hidden_sizes}
    return (build_separate_ff_actor(envs, in_size, hidden_sizes[PI_KEY], hidden_activation, continuous_parameterization=continuous_parameterization),
            build_separate_ff_critic(in_size, hidden_sizes[CRITIC_KEY], hidden_activation))


def build_separate_ff_option_critic(envs: Env,
                                   in_size: int,
                                   hidden_sizes: Union[Sequence[int], Dict[str, Sequence[int]]],
                                   hidden_activation: Callable[[], nn.Module] = Tanh,
                                   num_policies: int = 1,
                                   continuous_parameterization='beta', critic_in_size: Optional[int] = 0, option_in_size: Optional[int] = 0) -> Tuple[FFActor, FFFullNet, FFActor, FFActor]:
    """Actor, critic, termination, and option actor networks"""
    if isinstance(hidden_sizes, Sequence): hidden_sizes = {PI_KEY: hidden_sizes, CRITIC_KEY: hidden_sizes, TERMINATION_KEY: hidden_sizes, OPTION_ACTOR_KEY: hidden_sizes}
    return (
        build_separate_ff_actor(envs, in_size, hidden_sizes[PI_KEY], hidden_activation, num_policies, continuous_parameterization),
        build_separate_ff_critic(critic_in_size or in_size, hidden_sizes[CRITIC_KEY], hidden_activation, num_policies),
        build_separate_ff_termination(in_size, hidden_sizes[TERMINATION_KEY], hidden_activation, num_policies),
        build_separate_ff_option_actor(option_in_size or in_size, hidden_sizes[OPTION_ACTOR_KEY], hidden_activation, num_policies),
    )




