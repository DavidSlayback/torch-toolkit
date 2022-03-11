from typing import Dict, Union, Optional

import torch as th
import numpy as np

"""Commonly-used types"""
Tensor = th.Tensor
OptionalTensor = Optional[Tensor]
Array = np.ndarray
OptionalArray = Optional[Array]
XArray = Union[Tensor, Array]
OptionalXArray = Optional[XArray]
TensorDict = Dict[str, Tensor]
ArrayDict = Dict[str, Array]
XArrayDict = Dict[str, XArray]
