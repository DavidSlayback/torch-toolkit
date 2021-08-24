__all__ = ['a2c_pg_loss', 'a2c_vf_loss', 'ent_loss']

import torch as th
import torch.nn.functional as F
Tensor = th.Tensor


def a2c_pg_loss(advantages: Tensor, log_probs: Tensor):
    return (-advantages * log_probs).mean()


def a2c_vf_loss(returns: Tensor, values: Tensor):
    return F.mse_loss(values, returns, reduce=True, reduction='mean')


def ent_loss(entropies: Tensor):
    return (-entropies).mean()