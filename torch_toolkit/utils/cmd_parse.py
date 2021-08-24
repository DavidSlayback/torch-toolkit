__all__ = ['str_to_iterable', 'str_to_schedule', 'str_to_act']

from typing import Type, Iterable, Callable
import math


def str_to_iterable(string: str, return_type: Type = int, return_fn: Callable[[...], Iterable] = list) -> Iterable:
    string = string.strip()  # Remove outer whitespace
    delims = string[0] + string[-1]  # Surrounding context (i.e., (), [], <>, etc)
    if not delims.isnumeric(): string = string.strip(delims)  # Remove it
    vars = (s.strip() for s in string.split(', '))  # Split according to comma
    return return_fn(return_type(v) for v in vars)


def str_to_schedule(string: str, n_itr: int) -> Callable[[int], float]:
    if '>' in string:  # Linear initial>final*frac
        initial, string = string.split('>'); initial = float(initial) # First value is initial learning rate
        final, frac = (float(s) for s in string.split('*'))  # Final value, final fraction of iterations (i.e., proportion of iterations over which to decay)
        end_itr = min(int(n_itr * frac), n_itr)  # Iteration of last learning rate value
        lr_lam = lambda itr = 0: (1. - min(itr / end_itr, 1.)) * (initial - final) + final  # Fraction done * difference in values + end value
    elif '^' in string:  # Exponential value_from^value_to*frac
        initial, string = string.split('^'); initial = float(initial)
        final, frac = (float(s) for s in string.split('*'))
        end_itr = min(int(n_itr * frac), n_itr)
        b = math.log(final/initial) / (end_itr - 1)
        def lr_fn(itr: int) -> float:
            if itr == 0: return initial
            elif itr >= end_itr: return final
            else: return initial * math.exp(b * itr)
        lr_lam = lr_fn
    else:
        lr = float(string)
        lr_lam = lambda itr = 0: lr
    return lr_lam


def str_to_act(string: str):
    import torch.nn as nn
    if string == 'relu': return nn.ReLU
    elif string == 'elu': return nn.ELU
    elif string == 'tanh': return nn.Tanh
    elif string == 'sigmoid': return nn.Sigmoid
    else: raise ValueError("No valid activation")