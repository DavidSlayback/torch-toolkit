__all__ = ['str_to_iterable', 'str_to_schedule', 'str_to_act']

from typing import Type, Iterable, Callable, Optional
import math


def str_to_iterable(string: Optional[str], return_type: Type = int, return_fn: Callable[..., Iterable] = list) -> Optional[Iterable]:
    """Convert argument string into an interable

    Args:
        string: String (e.g., from argparse)
        return_type: Type of elements in iterable
        return_fn: Callable that converts elements to iterable. Defaults to "list"
    Returns:
        iterable: Iterable with elements
    """
    if string == 'None': return None
    string = string.strip()  # Remove outer whitespace
    delims = string[0] + string[-1]  # Surrounding context (i.e., (), [], <>, etc)
    if not delims.isnumeric(): string = string.strip(delims)  # Remove it
    varis = (s.strip() for s in string.split(','))  # Split according to comma
    return return_fn(return_type(v) for v in varis)


def str_to_schedule(string: str, n_itr: int) -> Callable[[int], float]:
    """Convert a specially-formatted string to a lambda schedule function based on training iterations

    Passing basic numbers (e.g., 1e-4) results in a constant schedule
    If string is of format a>b*c, results in a linear schedule from a -> b over the course of fraction c of total iterations
    If string is of format a^b*c, results in an exponential schedule from a -> b over the course of fraction c
    Args:
        string: Formatted string.
        n_itr: Total number of training iterations
    Returns:
        schedule: Callable function that takes current iteration as input to return current schedule value

    """
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


ns_act = {
    'relu': 'ReLU',
    'relu6': 'ReLU6',
    'swish': 'SiLU',
    'silu': 'SiLU',
    'prelu': 'PReLU',
    'rrelu': 'RReLU',
}


def str_to_act(string: str):
    """Get activation function corresponding to string. Some special-case handling for weirdly capitalized layers

    Args:
        string: String to parse. Something like 'relu', 'tanh', etc
    Returns:
        nn.Module class of corresponding activation. Raises error if it can't resolve one
    """
    import torch.nn.modules.activation as act
    for modification in (string, string.title(), string.upper(), string.lower()):
        """Attempt various modifications to string"""
        if hasattr(act, modification): return getattr(act, modification)
    try:
        """Check our nonstandard string dictionary"""
        return getattr(act, ns_act[string])
    except:
        raise ValueError("No valid activation found")