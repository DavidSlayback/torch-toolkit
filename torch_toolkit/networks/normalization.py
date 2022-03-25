__all__ = ['ObservationNormalizationModule']

import torch
import torch.nn as nn
Tensor = torch.Tensor

class ObservationNormalizationModule(nn.Module):
    """Standalone PyTorch module to normalize observations (without gradient)

    Args:
        n_obs: Number of observations from environment. Assumes vector
    """
    def __init__(self, n_obs: int):
        super().__init__()
        self.n_obs = n_obs
        self.num_steps = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.running_mean = nn.Parameter(torch.zeros(n_obs), requires_grad=False)
        self.running_variance = nn.Parameter(torch.zeros(n_obs), requires_grad=False)

    @torch.no_grad()
    def forward(self, x: Tensor, running_update: bool = False) -> Tensor:
        """Normalize observation. Optionally, update statistics before"""
        if running_update: self.update_normalization(x)
        return self.normalize(x)

    @torch.jit.export
    @torch.no_grad()
    def update_normalization(self, x: Tensor) -> None:
        """Update mean, variance, and steps"""
        xv = x.view(-1, self.n_obs)
        self.num_steps.add_(xv.shape[0])
        input_to_old_mean = xv - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=0)
        self.running_meana.add_(mean_diff)
        input_to_new_mean = xv - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=0)
        self.running_variance.add_(var_diff)

    @torch.jit.export
    def normalize(self, x: Tensor) -> Tensor:
        """Normalize input tensor with current statistics"""
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((x - self.running_mean) / variance.sqrt()).clip(-5, 5)