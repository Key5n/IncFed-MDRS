import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def plot_subsampling(
    X: list,
    Y_list: list[NDArray],
    stds_list: list[list[float]],
    labels: list[str],
    filename: str,
):
    X = np.array(X)

    for Y, stds, label in zip(Y_list, stds_list, labels):
        Y = np.array(Y)
        stds = np.array(stds)

        plt.plot(X, Y, marker="o", label=label)
        plt.fill_between(X, Y + stds, Y - stds, alpha=0.15)

    plt.xlabel("Subsampling size")
    plt.ylabel("Performance")
    plt.legend()

    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filename)

    plt.clf()
    plt.close()
