import matplotlib.pyplot as plt


def boxplot(X, tick_labels: list[str], filename: str):
    _, ax = plt.subplots()
    ax.boxplot(X, tick_labels=tick_labels, showfliers=False)
    ax.set_title("Evaluation Metrics")
    ax.set_ylabel("Metrics Score")

    plt.savefig(filename)

    plt.clf()
    plt.close()
