__all__ = ['vectorized_multinomial']

import numpy as np


def vectorized_multinomial(selected_prob_matrix: np.ndarray, random_numbers: np.ndarray) -> np.ndarray:
    """Vectorized sample from [B,N] probabilitity matrix

    Lightly edited from https://stackoverflow.com/a/34190035/2504700

    Args:
        selected_prob_matrix: (Batch, p) size probability matrix (i.e. T[s,a] or O[s,a,s']
        random_numbers: (Batch,) size random numbers from np.random.rand()
    Returns:
        (Batch,) size sampled integers
    """
    s = selected_prob_matrix.cumsum(axis=1)  # Sum over p dim for accumulated probability
    return (s < np.expand_dims(random_numbers, axis=-1)).sum(axis=1)  # Returns first index where random number < accumulated probability