__all__ = ['set_grad_enabled']

import sys
import torch
import functools
import inspect
from typing import Any, Callable, TypeVar, cast

class set_grad_enabled(torch.autograd.grad_mode._DecoratorContextManager):
    r"""Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This version is compatible with Torchscript. Pytorch version complains
        that no object is returned.
    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> with torch.no_grad():
        ...   y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False


    """
    def __init__(self, mode: bool):
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)