import matplotlib.pyplot as plt


def boxplot(X, tick_labels: list[str], colors: list[str], filename: str):
    _, ax = plt.subplots()
    bplot = ax.boxplot(X, tick_labels=tick_labels, showfliers=False)
    ax.set_title("Evaluation Metrics")
    ax.set_ylabel("Metrics Score")

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    plt.savefig(filename)

    plt.clf()
    plt.close()
