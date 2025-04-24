import matplotlib.pyplot as plt


def plot_line_graph(
    X: list, Y_list: list[list], label_list: list[str], filepath: str, xlabel: str
):
    plt.rcParams["font.size"] = 20
    plt.figure(figsize=(6, 6))
    for Y, label in zip(Y_list, label_list):
        plt.plot(X, Y, label=label, marker="o")

    plt.xlabel(xlabel)
    plt.ylabel("PATE")

    plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)

    plt.clf()
    plt.close()
