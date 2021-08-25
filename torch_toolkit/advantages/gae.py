__all__ = ['generalized_advantage_estimation', 'lambda_returns']

import torch as th
Tensor = th.Tensor


@th.jit.script
def generalized_advantage_estimation(v_tm1_t: Tensor, gamma_t: Tensor, r_t: Tensor, lambda_: float = 1., norm_adv: bool = True):
    """Generalized advantage estimate with λ-returns. Optionally normalize advantage

    Args:
        v_tm1_t: [T+1xB?] values of rollout and boostrap
        gamma_t: [TxB?] (episode done after step t signal * discount)
        r_t: [TxB?] rewards
        lambda_: Lambda mixing parameter. Default 1 results in basic discounted return
        norm_adv: Whether to normalize advantage as in PPO. Default True
    """
    v_tm1, v_t = v_tm1_t[:-1], v_tm1_t[1:]
    deltas = (r_t + gamma_t * v_t - v_tm1)
    adv = th.zeros_like(v_t)
    lastgaelam = th.zeros_like(v_t[0])
    for t in th.arange(v_t.shape[0] - 1, -1, -1, device=v_t.device):
        lastgaelam = adv[t] = deltas[t] + gamma_t[t] * lambda_ * lastgaelam
    ret = adv + v_tm1
    if norm_adv: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret

# @th.jit.script
def lambda_returns(v_t, gamma_t, r_t, lambda_: float = 1.):
    """Compute λ-returns on their own. v_t is t=1, includes bootstrap."""
    ret = th.cat((th.zeros_like(r_t), v_t[-1]), 0)
    inv_lambda = 1. - lambda_
    for t in th.arange(v_t.shape[0] - 1, -1, -1):
        ret[t] = r_t[t] + gamma_t[t] * (inv_lambda * v_t[t] + lambda_ * ret[t+1])
    return ret[:-1]