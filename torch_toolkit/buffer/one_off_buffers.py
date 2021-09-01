from functools import partial
from typing import Optional, Union, Dict

import torch as th

from ..utils.device import th_device, to_th, to_np, torch_type_to_np

Tensor = th.Tensor
import numpy as np

Array = np.ndarray
Arr = Union[Array, Tensor]
ArrDict = Dict[str, Arr]
Device = Union[str, th.device]

"""One-off buffer classes for different algorithms

Basic needs:
  Let's say batch size is B, and rollout length is T. Index from 0 (so steps are in half-open interval [0,T)).
  All buffers need to store observations, actions, rewards, and episode terminations. Call these o, a, r, d, vectors O,A,R,D
  Our first rollout will start with O[0] = o = env.reset()
  We typically do a final bootstrap step as well. Many approaches don't actually fold this into the rollout buffer
  But I want to. So O[T], _, _, _ = env.step(A[T-1]). This is also O[0] for next rollout
  Then _, V[T] = agent(O[T]) for bootstrap value. 
  So far, O and V need to be T+1 length. When we reset, we also need O[0] = O[-1]

Additional outputs:
  Actor-critic: logprob(a|o), v(o). Don't actually need these for basic actor-critic until grad
  PPO: logprob(a|o), v(o). Do need these for computing ratios and doing clipping.
  
Postprocessing:
  Advantages and returns need the bootstrap value and next done. So far, D[T-1] is next_done, V[T] is bootstrap
  PPO: Need minibatch sampling, but that can be easily done on O[:-1], A, R, Adv, Ret, V[:-1], lp
  
Recurrency considerations:
  If only extra input is state, and I'm only randomizing over the batch dimension (to preserve time), then I only need initial state
  Now I want a[t-1] as input (previous action). So A needs to be T+1 length. A[1:] recovers original. Same for reward.
  If I want to keep resets IN the network, not the algorithm, then I also need d[t-1], so D needs to be T+1 length. D[1:] recovers original
  Input at step t: LP[t], V[t], state = agent(O[t], state, A[t], R?[t], D[t]).
  Input at step T: _, V[T], _ = agent(O[T], state, A[T], R?[T], D[T]). So still easy
  
Option-critic considerations:
  Options are denoted as w. Need to track prev_w (for termination) and w (for action sampling). So W is T+1
  Values are now q(s,w). I need values for previous option and sampled option. Also average value v(s). Q[T+1, nopt]
  Can still just save logprobs of actions under sampled options, so LP[T, nact]
  
  
"""


class ActorCriticBuffer:
    # Which keys do we store extra
    prev_keys = ('a', 'd')  # prev action, is_inits
    next_keys = ('o', 'v')  # next obs, bsv

    def __init__(self,
                 T: int,
                 storage_device: Optional[Device] = None,
                 step_device: Device = 'cpu',
                 grad_device: Device = 'cuda'):
        self.device = th_device(storage_device)
        store_as_tensor = storage_device or False
        self.convert_for_storage = partial(to_th, device=self.device) if store_as_tensor else to_np
        self.convert_for_input = partial(to_th, device=th.device(step_device))
        self.convert_for_grad = partial(to_th, device=th.device(grad_device))
        self._T = T
        self._i = 0

    def init(self, init_o: Arr, prev_a: Arr):
        B = init_o.shape[0]
        self.o_b = self._buffer_from_example(init_o, True)
        self.o = self.o_b[:-1]
        self.a_b = self._buffer_from_example(prev_a, True)
        self.prev_a, self.a = self.a_b[:-1], self.a_b[1:]
        self.d_b = self._buffer_from_example(np.ones(B), True)
        self.prev_done, self.done = self.d_b[:-1], self.d_b[1:]
        self.r = self._buffer_from_example(np.zeros(B), False)
        self.v = self._buffer_from_example(np.zeros(B), True)
        return self._get_next_input()

    def _get_next_input(self):
        """Get current model input"""
        return self.convert_for_input(self.o[self._i], self.prev_a[self._i], self.prev_done[self._i])

    def _get_all(self):
        assert self._i == self._T
        return self.convert_for_grad(self.)

    def next_rollout(self):
        """Roll inputs over"""
        self.o_b[0] = self.o_b[-1]
        self.a_b[0] = self.a_b[-1]
        self.d_b[0] = self.d_b[-1]
        self._i = 0
        return self._get_next_input()

    def _buffer_from_example(self, x: Arr, extended: bool = False):
        dtype = torch_type_to_np(x.dtype)  # Get dtype so we can downcast float64 to float32 if needed
        if isinstance(dtype, (np.float64, np.bool)): dtype = np.float32  # Booleans and doubles to floats
        b_shape = (self._T + extended,) + tuple(x.shape)
        base_buffer = self.convert_for_storage(np.zeros(b_shape, dtype=dtype))  # Create buffer
        if extended: base_buffer[0] = self.convert_for_storage(x)  # Add initial obs

    def add(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k)[self._i] = self.convert_for_storage(v)

