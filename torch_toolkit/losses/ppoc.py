__all__ = ['ppoc_pg_loss', 'ppoc_vf_loss', 'ppoc_w_pg_loss', 'ppoc_beta_loss']

from typing import Optional

import torch as th
import torch.nn.functional as F

from .a2oc import a2oc_w_pg_loss, a2oc_vf_loss, a2oc_beta_loss

Tensor = th.Tensor


def ppoc_pg_loss(advantages: Tensor, new_log_probs: Tensor, old_log_probs: Tensor, clip_coef: float):
    """Single PPO epoch. Return clipped policy loss"""
    ratio = (new_log_probs - old_log_probs).exp()
    pi_loss1 = -advantages * ratio
    pi_loss2 = -advantages * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return th.maximum(pi_loss1, pi_loss2).mean()


def ppoc_vf_loss(returns: Tensor, new_q_sw: Tensor, old_q_sw: Tensor, clip_coef: float = 0.):
    """Single PPOC epoch. Return (potentially) clipped value loss. Don't clip if 0.

    Args:
        returns: Return targets
        new_q_sw: Value of sampled option from new net
        old_q_sw: Value of sampled option from old rollout
        clip_coef: PPO clipping. If 0, don't apply
    """
    if clip_coef:
        vf_loss_unclipped = F.mse_loss(new_q_sw, returns, reduction='none')  # Basic MSE loss, don't reduce
        v_clipped = old_q_sw + th.clamp(new_q_sw - old_q_sw, -clip_coef, clip_coef)
        vf_loss_clipped = F.mse_loss(v_clipped, returns, reduction='none')
        return th.maximum(vf_loss_clipped, vf_loss_unclipped).mean()
    else: return a2oc_vf_loss(returns, new_q_sw)


def ppoc_w_pg_loss(option_advantages: Tensor, new_option_log_probs_tp1: Tensor, old_option_log_probs_tp1: Tensor,
                   new_beta_tp1: Tensor, gamma: float, clip_coef: float = 0,
                   was_option_selected: Optional[Tensor] = None) -> Tensor:
    """Option-critic option policy loss

    Args:
        option_advantages: Advantages for option policy (G - V(s'))
        new_option_log_probs_tp1: Log probability of options (pi_w(w'|s'))
        old_option_log_probs_tp1: Log probability of options (pi_w(w'|s'))
        new_beta_tp1: Termination probability of sampled option in successor state (beta(s', w)). Use most recent
        gamma: Discount factor (loss is multiplied by this)
        clip_coef: PPO clipping coefficient
        was_option_selected: Whether we actually selected an option for each timestep (episode start OR option termination)
          If provided, learn only where we had to select an option.
    """
    if clip_coef:
        ratio = (new_option_log_probs_tp1 - old_option_log_probs_tp1).exp()
        if was_option_selected: ratio.mul_(was_option_selected)
        pi_loss1 = -option_advantages * ratio
        pi_loss2 = -option_advantages * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        l = th.maximum(pi_loss1, pi_loss2) * new_beta_tp1.detach()
        return l.mean().mul_(gamma)
    else: return a2oc_w_pg_loss(option_advantages, new_option_log_probs_tp1, new_beta_tp1, gamma, was_option_selected)


def ppoc_beta_loss(termination_advantages: Tensor, new_beta_stp1_w: Tensor, old_beta_stp1_w: Tensor,
                   gamma: float, delib_cost: float, clip_coef: float = 0., first_step: Optional[Tensor] = None) -> Tensor:
    """Option-critic option termination loss

    Args:
        termination_advantages: Termination advantage A(s', w) = Q(s', w) - V(s')
        new_beta_stp1_w: Termination probability of sampled option in successor state (beta(s', w))
        old_beta_stp1_w: Termination probability of sampled option in successor state (beta(s', w))
        gamma: Discount factor
        delib_cost: Termination regularizer added to loss
        first_step: Whether we were forced to terminate by episode boundary.
          If provided, learn only where we weren't forced to terminate
    """
    if clip_coef:
        ratio = (new_beta_stp1_w.log() - old_beta_stp1_w.log()).exp()  # Ratio for clipping
        if first_step: ratio.mul_(1. - first_step.float())
        A = termination_advantages + delib_cost
        beta_loss1 = A * ratio
        beta_loss2 = A * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        l = th.minimum(beta_loss1, beta_loss2)  # Min because advantages should be positive?
        return -l.mean().mul_(gamma)
    else: return a2oc_beta_loss(termination_advantages, new_beta_stp1_w, gamma, delib_cost, first_step)