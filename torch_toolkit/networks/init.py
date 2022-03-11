__all__ = ['layer_init', 'ORTHOGONAL_INIT_VALUES', 'ORTHOGONAL_INIT_VALUES_TORCH', 'pi_init', 'v_init', 'beta_init', 'rnn_init', 'module_init']

from typing import Dict

import torch.nn.init as init
import torch.nn as nn
from functools import partial

ORTHOGONAL_INIT_VALUES_TORCH = {
    nn.ReLU: 2. ** 0.5,  # He et. al 2015
    nn.ELU: 1.55 ** 0.5,  # https://stats.stackexchange.com/a/320443
    nn.SELU: 3. / 4,
    nn.Tanh: 5. / 3,
    nn.Sigmoid: 1.,
    nn.Linear: 1.,
    nn.Softmax: 1e-2
}
ORTHOGONAL_INIT_VALUES = {
    'relu': 2. ** 0.5,  # He et. al 2015
    'elu': 1.55 ** 0.5,  # https://stats.stackexchange.com/a/320443
    'selu': 3. / 4,
    'tanh': 5. / 3,
    'sigmoid': 1.,
    'linear': 1.,
    'pi': 1e-2
}


def layer_init(layer, std=ORTHOGONAL_INIT_VALUES['relu'], bias_const=0.0):
    """Orthogonal layer initialization with variable gain. Ignore LayerNorm

    Args:
        layer: nn.Module to be initialized
        std: Gain applied in orthogonal initialization
        bias_const: bias used for layers with bias
    """
    for n, p in layer.named_parameters():
        if n.startswith('ln'): continue
        if 'weight' in n: init.orthogonal_(p, std)
        else: init.constant_(p, bias_const)
    return layer


"""Common init schemes"""
rnn_init = beta_init = partial(layer_init, std=ORTHOGONAL_INIT_VALUES['sigmoid'])
pi_init = partial(layer_init, std=ORTHOGONAL_INIT_VALUES['pi'])
v_init = partial(layer_init, std=ORTHOGONAL_INIT_VALUES['linear'])

NORM_CLASSES = (nn.LayerNorm, nn.modules.batchnorm._NormBase, nn.GroupNorm)
def module_init(module: nn.Module,
                default_init_value: float = ORTHOGONAL_INIT_VALUES['relu'],
                specific_init: Dict[str, float] = {'actor_head': ORTHOGONAL_INIT_VALUES['pi'],
                                                   'critic_head': ORTHOGONAL_INIT_VALUES['linear'],
                                                   'termination_head': ORTHOGONAL_INIT_VALUES['sigmoid']},
                ignore_activation_gain: bool = True
                ) -> None:
    """Orthogonal layer initialization for an entire module

    Useful for reinitializing models (e.g., with different seed).
    Also useful if you want to use LazyModuleMixin

    Args:
        module: Full PyTorch Module instance to apply initialization to
        specific_init: Dict of k,v pairs. k corresponds to name of named modules, v is float value defining the gain for those modules
        ignore_activation_gain: If True, use default value for layer initialization gain, regardless of activation function.
          If False, attempt to match gain to the activation function following each layer. Not currently supported.
    """
    assert ignore_activation_gain  # Hopefully temporary
    for n, m in module.named_modules():
        if not n: continue  # Skip root
        if isinstance(m, nn.Embedding): m.reset_parameters()  # Standard initialization is fine
        elif isinstance(m, NORM_CLASSES):  # Norm classes start with weight 1., bias 0.
            for n2, p in m.named_parameters():
                if 'weight' in n2: nn.init.constant_(p, 1.)
                else: nn.init.constant_(p, 0.)
        elif isinstance(m, (nn.RNNBase, nn.RNNCellBase)):  # Recurrent classes
            layer_init(m, ORTHOGONAL_INIT_VALUES['tanh'])
        else:
            gain = default_init_value
            for k, v in specific_init.items(): # Use our keys to switch gain
                if k in n: gain = v
            for n2, p in m.named_parameters(recurse=False):
                if 'weight' in n2: nn.init.orthogonal_(p, gain)
                else: nn.init.constant_(p, 0.)  # Handles both learnable initial states and biases


