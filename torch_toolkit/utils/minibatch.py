__all__ = ['iterate_mb_idxs']

import numpy as np

def iterate_mb_idxs(data_length, minibatch_size, shuffle=False):
    """Yields minibatches of indexes, to use as a for-loop iterator, with
    option to shuffle.
    """
    if shuffle:
        indexes = np.arange(data_length)
        np.random.shuffle(indexes)
    for start_idx in range(0, data_length - minibatch_size + 1, minibatch_size):
        batch = slice(start_idx, start_idx + minibatch_size)
        if shuffle:
            batch = indexes[batch]
        yield batch