import numpy as np
from numpy.typing import NDArray


def subsample(x: NDArray, subsampling_size: int) -> NDArray:
    x = np.reshape(x, (-1, 1))

    x_subsampled = x[:subsampling_size]

    return x_subsampled
