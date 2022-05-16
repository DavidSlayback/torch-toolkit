__all__ = ['a2oc_pg_loss', 'a2oc_vf_loss', 'a2oc_w_pg_loss', 'a2oc_beta_loss', 'w_ent_loss']
from typing import Optional

import torch as th
import torch.nn.functional as F

Tensor = th.Tensor


def a2oc_pg_loss(advantages: Tensor, log_probs: Tensor) -> Tensor:
    """Option-critic primitive policy loss. No special treatment, only saving logprobs for actual sampled option"""
    return (-advantages * log_probs).mean()


def a2oc_vf_loss(returns: Tensor, q_sw: Tensor) -> Tensor:
    """Option-critic critic loss."""
    return F.mse_loss(q_sw, returns, reduce=True, reduction='mean')


def a2oc_w_pg_loss(option_advantages: Tensor, option_log_probs_tp1: Tensor, beta_tp1: Tensor,
                   gamma: float, was_option_selected: Optional[Tensor] = None) -> Tensor:
    """Option-critic option policy loss

    Args:
        option_advantages: Advantages for option policy (G - V(s'))
        option_log_probs_tp1: Log probability of options (pi_w(w'|s'))
        beta_tp1: Termination probability of sampled option in successor state (beta(s', w))
        sampled_option: Sampled option during rollout
        gamma: Discount factor (loss is multiplied by this)
        was_option_selected: Whether we actually selected an option for each timestep (episode start OR option termination)
          If provided, learn only where we had to select an option.
    """
    l = (-option_advantages * option_log_probs_tp1 * beta_tp1.detach())
    if was_option_selected is not None: l.mul_(was_option_selected.float())
    return l.mean().mul_(gamma)


def w_ent_loss(option_entropies: Tensor, option_selected: Optional[Tensor] = None) -> Tensor:
    """Entropy loss with optional masking"""
    l = -option_entropies
    if option_selected is not None: l.mul_(option_selected.float())
    return l.mean()


def a2oc_beta_loss(termination_advantages: Tensor, beta_tp1: Tensor,
                   gamma: float, delib_cost: float, first_step: Optional[Tensor] = None) -> Tensor:
    """Option-critic option termination loss

    Args:
        termination_advantages: Termination advantage A(s', w) = Q(s', w) - V(s')
        beta_tp1: Termination probability of sampled option in successor state (beta(s', w))
        gamma: Discount factor
        delib_cost: Termination regularizer added to loss
        first_step: Whether we were forced to terminate by episode boundary.
          If provided, learn only where we weren't forced to terminate
    """
    l = (termination_advantages + delib_cost) * beta_tp1
    if first_step: l.mul_(1 - first_step.float())
    return -l.mean().mul_(gamma)

