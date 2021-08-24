__all__ = ['normalize']

import torch as th
Tensor = th.Tensor


@th.jit.script
def normalize(x: Tensor):
    return (x - x.mean()) / (x.std() + 1e-8)