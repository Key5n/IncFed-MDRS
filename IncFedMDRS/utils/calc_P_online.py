import numpy as np
from numpy.typing import NDArray


def calc_P_online(precision_matrix, x, lam: float = 1, update: int = 1):
    x = np.reshape(x, (-1, 1))
    next_precision_matrix = precision_matrix
    for _ in np.arange(update):
        gain = 1 / lam * np.dot(next_precision_matrix, x)
        gain = gain / (1 + 1 / lam * np.dot(np.dot(x.T, next_precision_matrix), x))
        next_precision_matrix = (
            1
            / lam
            * (next_precision_matrix - np.dot(np.dot(gain, x.T), next_precision_matrix))
        )
    return next_precision_matrix


def calc_P_online_based_on_pca(P: NDArray, x: NDArray, eigenvalue: float) -> NDArray:
    x = np.reshape(x, (-1, 1))

    numerator = eigenvalue * np.dot(np.dot(np.dot(P, x), x.T), P)
    denominator = 1 + eigenvalue * np.dot(np.dot(x.T, P), x)
    next_P = P - numerator / denominator

    return next_P


def woodbury(A_inversed: NDArray, B: NDArray, C: NDArray) -> NDArray:
    _, k = B.shape

    tmp1 = np.dot(A_inversed, B)
    tmp2 = np.linalg.inv(np.eye(k) + np.dot(np.dot(C, A_inversed), B))
    tmp3 = np.dot(C, A_inversed)

    res = A_inversed - np.dot(np.dot(tmp1, tmp2), tmp3)

    return res
