__all__ = ['layer_init', 'ORTHOGONAL_INIT_VALUES', 'ORTHOGONAL_INIT_VALUES_TORCH', 'pi_init', 'v_init', 'beta_init', 'rnn_init']

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