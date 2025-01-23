import numpy as np
import matplotlib.pyplot as plt


def plot_subsampling(X, Y, stds, filename):
    X = np.array(X)
    Y = np.array(Y)
    stds = np.array(stds)

    plt.plot(X, Y, marker="o")
    plt.xlabel("Subsampling size")
    plt.ylabel("Performance")

    plt.ylim(0, 1)

    plt.fill_between(X, Y + stds, Y - stds, alpha=0.15)

    plt.tight_layout()
    plt.savefig(filename)

    plt.clf()
    plt.close()
