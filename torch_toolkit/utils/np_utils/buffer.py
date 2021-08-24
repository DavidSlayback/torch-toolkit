__all__ = ['NpBuffer']

import numpy as np


class NpBuffer:
    """Numpy circular buffer

    Args:
        max_length: Maximum length of numpy buffer
        dtype: Dtype of numpy buffer
    """
    __slots__ = ('index', 'buffer', 'max_length')

    def __init__(self, max_length: int = 100, dtype: np.dtype = np.float64):
        self.index = 0
        self.buffer = np.full(max_length, np.nan, dtype=dtype)
        self.max_length = max_length

    def __add__(self, other: np.ndarray):
        """Add numpy array to deque"""
        B = other.size
        if B:
            idxes = np.arange(self.index, self.index + B)
            idxes[idxes >= self.max_length] -= self.max_length  # Roll over
            self.buffer[idxes] = other  # Store
            self.index = (idxes[-1] + 1) % self.max_length  # Update start index
        return self

    def clear(self):
        self.index = 0
        self.buffer[:] = np.nan

    def __len__(self):
        return np.sum(~np.isnan(self.buffer))