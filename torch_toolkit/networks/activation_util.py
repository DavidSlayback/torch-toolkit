__all__ = ['maybe_inplace']

from typing import Callable

import torch.nn as nn


def maybe_inplace(act_fn: Callable[[...], nn.Module]) -> Callable[[...], nn.Module]:
    """Use in-place version of activation function if there is one

    Args:
        act_fn: nn.Module class such as nn.ReLU or nn.Tanh
    Returns:
        act_fn partial method
    """
    from functools import partial
    from inspect import signature
    if 'inplace' in signature(act_fn).parameters.keys(): act_fn = partial(act_fn, inplace=True)
    return act_fn