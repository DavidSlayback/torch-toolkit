__all__ = ['ReturnScaledTD']

import torch as th
import torch
Tensor = th.Tensor


class ReturnScaledTD(th.nn.Module):
    """Class that maintains running statistics for https://arxiv.org/abs/2105.05347v1. Outputs sigma which can be used to rescale TD-error

    Args:
        v_init: Value layer initialization, used as a minimum deviation
    """
    def __init__(self, v_init: float = 1.):
        super().__init__()
        self.register_buffer('min_v', th.tensor(v_init, dtype=th.float64))  # Minimum scale applied when we receive a new signal. Noise of linear value layer
        # self.min_v = Tensor(v_init).double()
        self.register_buffer('count', th.zeros((), dtype=th.int64))  # Count
        self.register_buffer('r_mean', th.zeros((), dtype=th.float64))  # Reward mean
        self.register_buffer('r_M2', th.ones((), dtype=th.float64))  # Reward 2nd moment
        self.register_buffer('gamma_mean', th.zeros((), dtype=th.float64))  # Discount mean
        self.register_buffer('gamma_M2', th.ones((), dtype=th.float64))  # Discount 2nd moment
        self.register_buffer('G2_mean', th.zeros((), dtype=th.float64))  # Mean squared return

    @th.jit.export
    def update(self, r_t: Tensor, gamma_t: Tensor, return_t: Tensor):
        """gamma_t is (1-d_t) * gamma (0 on episode boundary, gamma o.w.)"""
        self.count.add_(gamma_t.numel())  # Handles both T and B dimensions
        # Update all means
        delta_r = r_t - self.r_mean
        self.r_mean.add_(delta_r.sum() / self.count)
        delta_gamma = gamma_t - self.gamma_mean
        self.gamma_mean.add_(delta_gamma.sum() / self.count)
        delta_G = return_t ** 2 - self.G2_mean
        self.G2_mean.add_(delta_G.sum() / self.count)
        # Update M2 using new mean
        delta2_r = r_t - self.r_mean
        self.r_M2.add_((delta_r * delta2_r).sum())
        delta2_gamma = gamma_t - self.gamma_mean
        self.gamma_M2.add_((delta_gamma * delta2_gamma).sum())

    @th.no_grad()
    def forward(self, r_t: Tensor, gamma_t: Tensor, return_t: Tensor):
        """Return the scale sigma = sqrt(V[r] + V[gamma] * E[G**2]). Use variance, not sample variance"""
        self.update(r_t, gamma_t, return_t)  # Do stat updates first
        sigma = ((self.r_M2 / self.count) + (self.gamma_M2 / self.count) * self.G2_mean).sqrt()  # Overall scaling factor
        batch_sigma = (r_t.var() + gamma_t.var() * (return_t ** 2).mean()).sqrt()  # Batch scaling factor
        return torch.max(torch.stack((sigma, batch_sigma, self.min_v))).float().cpu().item()  # Return as float for easy uses