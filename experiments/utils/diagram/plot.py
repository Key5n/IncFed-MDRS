import matplotlib.pyplot as plt
from numpy.typing import NDArray


def range_convers_new(label):
    """
    input: arrays of binary values
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    """
    L = []
    i = 0
    j = 0
    while j < len(label):
        # print(i)
        while label[i] == 0:
            i += 1
            if i >= len(label):
                break
        j = i + 1
        # print('j'+str(j))
        if j >= len(label):
            if j == len(label):
                L.append((i, j - 1))

            break
        while label[j] != 0:
            j += 1
            if j >= len(label):
                L.append((i, j - 1))
                break
        if j >= len(label):
            break
        L.append((i, j - 1))
        i = j
    return L


def plot(scores: NDArray, labels: NDArray, filename: str):
    range_anomaly = range_convers_new(labels)

    # # Create a figure and axis
    _, ax = plt.subplots(figsize=(10, 6))

    # Plot the anomaly scores
    ax.plot(scores, label="Anomaly Score", color="blue", linestyle="-")

    for r in range_anomaly:
        ax.axvspan(r[0], r[1], color="red", alpha=0.5)

    # Add grid, labels, and title
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Anomaly Score", fontsize=12)

    # Add a legend
    ax.legend(loc="upper left", fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.savefig(filename)

    plt.clf()
    plt.close()
