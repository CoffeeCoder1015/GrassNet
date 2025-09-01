# Batch generator
import numpy as np


def get_batches(X, Y, batch_size=100, shuffle = True):
    """
    Yielding generator of batches (X,Y) test data at `batch_size` per call
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]