__all__ = ['generalized_advantage_estimation_np', 'lambda_returns_np']

from typing import Tuple
import numpy as np
Array = np.ndarray
from numba import njit


@njit
def generalized_advantage_estimation_np(v_tm1_t: Array, gamma_t: Array, r_t: Array,
                                        lambda_: float = 1., norm_adv: bool = False) -> Tuple[Array, Array]:
    """Generalized advantage estimate with λ-returns. Optionally normalize advantage

    Args:
        v_tm1_t: [T+1xB?] values of rollout and boostrap
        gamma_t: [TxB?] (episode done after step t signal * discount)
        r_t: [TxB?] rewards
        lambda_: Lambda mixing parameter. Default 1 results in basic discounted return
        norm_adv: Whether to normalize advantage as in PPO. Default False
    Returns:
        adv: A_t for t=0->T, computed with GAE
        ret: λ-returns for t=0->T
    """
    v_tm1, v_t = v_tm1_t[:-1], v_tm1_t[1:]
    deltas = (r_t + gamma_t * v_t - v_tm1)
    gamlam_t = gamma_t * lambda_
    adv = np.zeros_like(r_t)
    lastgaelam = np.zeros_like(v_t[0])  # Batch size or scalar
    for i in range(r_t.shape[0] - 1, -1, -1):
        lastgaelam = adv[i] = deltas[i] + gamlam_t[i] * lastgaelam
    ret = adv + v_tm1
    if norm_adv: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


@njit
def lambda_returns_np(v_t: Array, gamma_t: Array, r_t: Array, lambda_: float = 1.) -> Array:
    """Compute λ-returns on their own.

    Args:
        v_t: [TxB?] values of rollout from step 1 including bootstrap
        gamma_t: [TxB?] episode done after step t signal * discount
        r_t: [TxB?] rewards
        lambda_: Lambda mixing parameter. Default 1 results in basic discounted return
    Returns:
        ret: λ-returns for t=0->T
    """
    inv_lambda = 1. - lambda_
    g = v_t[-1]
    ret = np.zeros_like(r_t)
    for t in range(v_t.shape[0] - 1, -1, -1):
        g = ret[t] = r_t[t] + gamma_t[t] * (inv_lambda * v_t[t] + lambda_ * g)
    return ret



