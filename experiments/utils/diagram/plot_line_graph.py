import matplotlib.pyplot as plt


def plot_line_graph(X: list, Y_list: list[list], label_list: list[str], filepath: str):
    for Y, legend in zip(Y_list, label_list):
        plt.plot(X, Y, label=legend)

    plt.xlabel("Client Size")
    plt.ylabel("Performance")

    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filepath)

    plt.clf()
    plt.close()
