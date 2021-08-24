__all__ = ['ppo_pg_loss', 'ppo_vf_loss']

import torch as th
import torch.nn.functional as F
Tensor = th.Tensor


def ppo_pg_loss(advantages: Tensor, new_log_probs: Tensor, old_log_probs: Tensor, clip_coef: float):
    """Single PPO epoch. Return clipped policy loss"""
    ratio = (new_log_probs - old_log_probs).exp()
    pi_loss1 = -advantages * ratio
    pi_loss2 = -advantages * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return th.max(pi_loss1, pi_loss2).mean()


def ppo_vf_loss(returns: Tensor, new_values: Tensor, old_values: Tensor, clip_coef: float = 0.):
    """Single PPO epoch. Return (potentially) clipped value loss. Don't clip if 0."""
    vf_loss_unclipped = F.mse_loss(new_values, returns, reduction='none')   # Basic MSE loss, don't reduce
    if clip_coef:
        v_clipped = old_values + th.clamp(new_values - old_values, -clip_coef, clip_coef)
        vf_loss_clipped = F.mse_loss(v_clipped, returns, reduction='none')
        return th.max(vf_loss_clipped, vf_loss_unclipped).mean()
    else: return vf_loss_unclipped.mean()