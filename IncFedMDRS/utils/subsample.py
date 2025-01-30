import numpy as np
from numpy.typing import NDArray


def subsample(x: NDArray, subsampling_size: int, seed: int) -> NDArray:
    rng = np.random.default_rng(seed)
    x_subsampled = rng.choice(x, subsampling_size, replace=False)

    return x_subsampled
