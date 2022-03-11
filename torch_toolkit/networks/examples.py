from typing import Iterable, Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
Tensor = th.Tensor

from .rnn import ResetGRU
from .mlp import MLP


class BasePOMDPBody(nn.Module):
    """Basic network body architecture for """
    def __init__(self,
                 obs_n: int,
                 act_n: int = 0,
                 embedding_dim: int = 128,
                 gru_size: int = 256,
                 gru_init_state_learnable: int = 0,
                 gru_layer_norm: bool = False,
                 hidden_sizes: Iterable[int] = (256,),
                 hidden_activations: nn.Module = nn.Tanh
                 ):
        super().__init__()
        self.embed = nn.Embedding(obs_n, embedding_dim)
        self.use_gru = gru_size > 0
        self.embed_previous_action = act_n
        if self.use_gru: self.gru = ResetGRU(embedding_dim + act_n, gru_size, gru_init_state_learnable, gru_layer_norm)
        self.body = MLP(gru_size or embedding_dim, list(hidden_sizes), hidden_activations)
        self._dim = self.body._dim

    def forward(self, x: Tensor, state: Optional[Tensor] = None,
                prev_action: Optional[Tensor] = None, reset: Optional[Tensor] = None, idx: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        ndim = x.dim()
        if ndim == 1:
            """Single-step rollout"""
            x = self.embed(x)
            if self.use_gru:
                if self.embed_previous_action and prev_action is not None: x = self._embed(x, prev_action)
                x, state = self.gru(x, state, reset, idx)
            return self.body(x), state
        else: return self._unroll(x, state, prev_action, reset)

    def _unroll(self, x: Tensor, state: Optional[Tensor] = None,
                prev_action: Optional[Tensor] = None, reset: Optional[Tensor] = None, idx: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Multi-step rollout. Embedding, then concat previous action, then gru, then body"""
        T, B = x.shape
        x = self.embed(x.view(T*B, -1)).view(T, B, -1)
        if self.use_gru:
            if self.embed_previous_action and prev_action is not None: x = self._embed(x, prev_action)
            x, state = self.gru(x, state, reset, idx)
        x = self.body(x.view(T*B, -1)).view(T, B, -1)
        return x, state

    def _embed(self, x: Tensor, prev_action: Tensor) -> Tensor:
        return th.cat((x, F.one_hot(prev_action, self.embed_previous_action)), -1)




