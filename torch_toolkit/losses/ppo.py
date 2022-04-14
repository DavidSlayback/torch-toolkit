__all__ = ['clipped_ppo_loss', 'unclipped_ppo_loss', 'kl_loss', 'ppo_vf_loss']

from typing import Optional

import torch as th
import torch.nn.functional as F
Tensor = th.Tensor


def clipped_ppo_loss(advantages: Tensor, logratio: Tensor, clip_coef: float):
    """Single PPO epoch. Return clipped policy loss"""
    ratio = logratio.exp()
    pi_loss1 = -advantages * ratio
    pi_loss2 = -advantages * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return th.maximum(pi_loss1, pi_loss2).mean()


def unclipped_ppo_loss(advantages: Tensor, logratio: Tensor, clip_coef: Optional[float] = 0.):
    """Single PPO epoch. Return unclipped policy loss"""
    ratio = logratio.exp()
    surr_loss = (-advantages * ratio).mean()
    return surr_loss


def kl_loss(logratio: Tensor, kl_coef: float, reverse: bool = False) -> Tensor:
    """Single PPO epoch. Return kl penalty"""
    return (logratio * logratio.exp()).mean() * kl_coef if not reverse else (-logratio).mean()


def ppo_vf_loss(returns: Tensor, new_values: Tensor, old_values: Tensor, clip_coef: float = 0.):
    """Single PPO epoch. Return (potentially) clipped value loss. Don't clip if 0."""
    vf_loss_unclipped = F.mse_loss(new_values, returns, reduction='none')   # Basic MSE loss, don't reduce
    if clip_coef:
        v_clipped = old_values + th.clamp(new_values - old_values, -clip_coef, clip_coef)
        vf_loss_clipped = F.mse_loss(v_clipped, returns, reduction='none')
        return th.maximum(vf_loss_clipped, vf_loss_unclipped).mean()
    else: return vf_loss_unclipped.mean()