import matplotlib.pyplot as plt


def boxplot(X, filename: str):
    _, ax = plt.subplots()
    ax.boxplot(X, showfliers=False)

    plt.savefig(filename)

    plt.clf()
    plt.close()
