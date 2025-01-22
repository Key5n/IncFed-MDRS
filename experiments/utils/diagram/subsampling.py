import matplotlib.pyplot as plt


def plot_subsampling(X, Y, filename):
    plt.plot(X, Y, marker="o")
    plt.xlabel("Subsampling size")
    plt.ylabel("Performance")

    plt.tight_layout()
    plt.savefig(filename)

    plt.clf()
    plt.close()
