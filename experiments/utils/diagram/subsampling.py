import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def plot_subsampling(
    X: list,
    Y_list: list[NDArray],
    labels: list[str],
    filename: str,
):
    plt.rcParams["font.size"] = 20
    plt.figure(figsize=(6, 6))
    X = np.array(X)

    for Y, label in zip(Y_list, labels):
        Y = np.array(Y)
        plt.plot(X, Y, marker="o", label=label)

    plt.xlabel("Subsampling Size")
    plt.ylabel("Performance")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)

    plt.clf()
    plt.close()
