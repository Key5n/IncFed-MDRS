import os
import matplotlib.pyplot as plt


def boxplot(X, tick_labels: list[str], colors: list[str], result_dir: str):
    _, ax = plt.subplots()
    ax.set_title("Evaluation Metrics")
    ax.set_ylabel("Metrics Score")

    bplot = ax.boxplot(X, tick_labels=tick_labels, showfliers=False, patch_artist=True)

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)

    filename = os.path.join(result_dir, "boxplot.png")
    plt.savefig(filename)

    plt.clf()
    plt.close()
