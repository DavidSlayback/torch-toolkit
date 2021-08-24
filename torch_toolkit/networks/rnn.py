__all__ = ['break_grad', 'mask_state', 'update_state_with_index', 'ResetGRU', 'LayerNormGRUCell']
from typing import Optional, Tuple

import torch as th
import torch.nn as nn
Tensor = th.Tensor
from .init import layer_init, ORTHOGONAL_INIT_VALUES


def break_grad(x: Tensor) -> Tensor:
    """Detach and reattach gradient (e.g., across batch boundaries)"""
    return x.detach_().requires_grad_()


def mask_state(state_original: Tensor, reset: Tensor, initial_state: Tensor) -> Tensor:
    return state_original * (1. - reset.unsqueeze(-1)) + initial_state.expand_as(state_original) * reset.unsqueeze(-1)


def update_state_with_index(state: Tensor, tidx: Tensor, ntidx: Tensor, idx_state: Tensor) -> Tensor:
    """Update state at idx with idx_state, otherwise return state"""
    s = th.zeros_like(state)
    s[tidx] = idx_state
    s[ntidx] = state[ntidx]
    return s


class LayerNormGRUCell(nn.RNNCellBase):
    """Layer-normalized GRU as in https://arxiv.org/pdf/1607.06450.pdf

    https://github.com/pytorch/pytorch/issues/12482#issuecomment-440485163"""
    def __init__(self, input_size, hidden_size, bias=True, ln_preact=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)
        self.ln_preact = ln_preact
        if ln_preact:
            self.ln_ih = nn.LayerNorm(3 * self.hidden_size)
            self.ln_hh = nn.LayerNorm(3 * self.hidden_size)
        self.ln_in = nn.LayerNorm(self.hidden_size)
        self.ln_hn = nn.LayerNorm(self.hidden_size)

    def forward(self, input: Tensor, hx: Tensor):
        ih = input @ self.weight_ih.t() + self.bias_ih
        hh = hx @ self.weight_hh.t() + self.bias_hh
        if self.ln_preact:
            ih = self.ln_ih(ih)
            hh = self.ln_hh(hh)

        i_r, i_z, i_n = ih.chunk(3, dim=1)
        h_r, h_z, h_n = hh.chunk(3, dim=1)
        i_n = self.ln_in(i_n)
        h_n = self.ln_hn(h_n)

        r = th.sigmoid(i_r + h_r)
        z = th.sigmoid(i_z + h_z)
        n = th.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * hx
        return h


class ResetGRU(nn.Module):
    """GRU that automatically resets state according to dones. Optional learnable state

    Args:
        input_size: Size of incoming tensor
        hidden_size: Size of GRU
        learnable_state: Whether to do a learnable initial state or just reset to 0s. If >1, multiple learnable states
        layer_norm: Whether to use a layer normalized GRUCell
    """
    def __init__(self, input_size: int, hidden_size: int, learnable_state: int = 0, layer_norm: bool = False):
        super().__init__()
        gru_cls = LayerNormGRUCell if layer_norm else nn.GRUCell
        self.core = layer_init(gru_cls(input_size, hidden_size), ORTHOGONAL_INIT_VALUES['sigmoid'])
        init_state = th.zeros(1, hidden_size)
        if learnable_state:
            init_state = th.tile(init_state, (learnable_state, 1))
            self.register_parameter('init_state', nn.Parameter(init_state))
        else: self.register_buffer('init_state', init_state)

    def forward(self, x: Tensor, state: Optional[Tensor] = None, reset: Optional[Tensor] = None, idx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass

        Args:
            x: Tensor input. Size is [T?xBxi].
            state: Tensor input for incoming state. Size is [Bxh]. If None, use initial state
            reset: Tensor input telling where to reset to initial state. Size if [T?xB]. If None, assume no resets.
            idx: Tensor input telling which initial state to use (if resetting). If None, assume first index
        """
        if idx is None: idx = th.zeros(x.shape[:-1], device=x.device, dtype=th.int64)
        if state is None: state = self.init_state[0].unsqueeze(0).expand(x.shape[-2], -1)
        if x.dim() == 2:  # Batch-only
            if reset is not None:
                state = mask_state(state, reset, self.init_state[idx])
            f = self.core(x, state)
            state = f.clone()  # State is separate
        else:  # Unroll
            T, B = x.shape[:2]
            states = []
            for t in range(T):
                if reset is not None: state = mask_state(state, reset[t], self.init_state[idx[t]])
                states.append(self.core(x[t], state))
                state = states[-1]
            f = th.stack(states, 0)
        return f, state

