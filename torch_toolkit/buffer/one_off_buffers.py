from typing import Optional, Union, Dict, Iterable, Any, Tuple
from functools import partial

from ..utils.device import th_device, to_th, to_np, torch_type_to_np

import torch as th
Tensor = th.Tensor
import numpy as np
Array = np.ndarray
Arr = Union[Array, Tensor]
ArrDict = Dict[str, Arr]
Device = Union[str, th.device]

"""One-off buffer classes for different algorithms"""

class ActorCriticBuffer:
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