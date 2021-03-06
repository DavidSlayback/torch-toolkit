__all__ = ['sample_discrete', 'sample_discrete_option', 'sample_continuous', 'sample_sac_continuous', 'sample_bernoulli', 'sample_continuous_beta']

import math
from typing import Optional

import torch
from torch.distributions.utils import _standard_normal
from torch.nn.functional import binary_cross_entropy_with_logits

from ..typing import Tensor
from ..utils import batched_index

"""Jit-compatible action sampling"""
def sample_discrete(logits: Tensor, action: Optional[Tensor] = None):
    """Return action, logprob, entropy, probs. If action is not provided, we sample one"""
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = torch.softmax(logits, -1)
    if action is None:
        with torch.no_grad(): action = torch.multinomial(probs, 1)
    else: action = action.unsqueeze(-1)
    log_prob = logits.gather(-1, action).squeeze(-1)
    entropy = -(logits * probs).sum(-1)
    return action.squeeze(-1), log_prob, entropy, probs


def sample_bernoulli(logits: Tensor, prev_option: Optional[Tensor] = None, termination: Optional[Tensor] = None):
    """Return termination"""
    probs = torch.sigmoid(logits)
    if prev_option is not None:
        logits = batched_index(prev_option, logits)  # Select to just previous option
    probs_w = torch.sigmoid(logits)
    if termination is None:
        with torch.no_grad(): termination = torch.bernoulli(probs_w)
    entropy = binary_cross_entropy_with_logits(logits, probs_w, reduction='none')
    lp = -binary_cross_entropy_with_logits(logits, termination, reduction='none')
    return termination, lp, entropy, probs


def sample_discrete_option(logits: Tensor, termination: Optional[Tensor] = None, prev_option: Optional[Tensor] = None, option: Optional[Tensor] = None, ):
    """Return option, logprob, entropy, probs. If option is not provided, we sample one.
    If termination and prev_option are provided, we replace terminated prev_options with new option"""
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = torch.softmax(logits, -1)
    if option is None:
        with torch.no_grad():
            option = torch.multinomial(probs, 1).squeeze(-1)  # Sample full set of new options
            if termination is not None and prev_option is not None:
                option = torch.where(termination > 0, option, prev_option)  # Replace previous options where terminated
    log_prob = logits.gather(-1, option.unsqueeze(-1)).squeeze(-1)
    entropy = -(logits * probs).sum(-1)
    return option, log_prob, entropy, probs.detach()


def sample_continuous(mean: Tensor, log_std: Tensor, action: Optional[Tensor] = None):
    """Return action, logprob, entropy, probs. If action is not provided, we sample one"""
    std = log_std.exp()
    if action is None:
        with torch.no_grad(): action = torch.normal(mean, std)
    var = std ** 2
    log_prob = (-((action - mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))).sum(-1)
    entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)
    return action, log_prob, entropy, log_prob.exp()


def sample_continuous_beta(alpha_softplus_one: Tensor, beta_softplus_one: Tensor, action_01: Optional[Tensor] = None):
    """Sample continuous action using beta distribution parameterized by alpha and beta vectors

    Args:
        alpha_softplus_one: Alpha vector, after softplus and +1
        beta_softplus_one: Beta vector, after softplus and +1
        action_01: Unscaled action
    """
    conc = torch.stack([alpha_softplus_one, beta_softplus_one], -1)
    if action_01 is None:
        with torch.no_grad(): action_01 = torch._sample_dirichlet(conc).select(-1, 0)
    ht = torch.stack([action_01, 1.0 - action_01], -1)
    a0 = conc.sum(-1)
    la1 = torch.lgamma(conc).sum(-1)
    # https://github.com/pytorch/pytorch/blob/9ad0578c590f2b698b52813243e39af94282a472/torch/distributions/beta.py#L63
    # https://github.com/pytorch/pytorch/blob/905efa82ff698f252a44fac83efaa72fe5678f52/torch/distributions/dirichlet.py#L67
    lp = (torch.log(ht) * (conc - 1)).sum(-1) + torch.lgamma(a0) - la1
    # https://github.com/pytorch/pytorch/blob/905efa82ff698f252a44fac83efaa72fe5678f52/torch/distributions/dirichlet.py#L84
    k = conc.size(-1)
    ent = (la1 - torch.lgamma(a0) - (k - a0) * torch.digamma(a0) -
                ((conc - 1.0) * torch.digamma(conc)).sum(-1))
    return action_01, lp.sum(-1), ent.sum(-1), lp.exp()


def sample_sac_continuous(mean: Tensor, log_std: Tensor, action_scale: Tensor,
                          action_bias: Tensor, action: Optional[Tensor], LOG_STD_MIN: float = -2, LOG_STD_MAX: float = 5.):
    """Return action, logprob, mean. Uses SAC-style squashing for log_std"""
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)  # From SpinUp / Denis Yarats
    std = log_std.exp()
    # Replacement for rsample
    x_t = mean + _standard_normal(std.shape, dtype=std.dtype, device=std.th_device) * std  # Uniform normal rescaled
    y_t = torch.tanh(x_t)
    if action is None:
        with torch.no_grad(): action = y_t * action_scale + action_bias
    var = std ** 2
    log_prob = (-((x_t - mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))).sum(-1)
    # Enforce action bound
    log_prob -= (action_scale * (1 - y_t ** 2) + 1e-6).log()
    log_prob = log_prob.sum(-1)
    mean = torch.tanh(mean) * action_scale + action_bias
    return action, log_prob, mean
