import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot(anomaly_scores: NDArray, true_labels: NDArray, filename: str):

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(10, 6))

    # Plot the anomaly scores
    ax.plot(
        anomaly_scores, label="Anomaly Scores", color="blue", linestyle="-", marker="o"
    )

    # Highlight true labels as a scatter plot
    anomalies = np.where(true_labels == 1)[0]
    ax.scatter(
        anomalies,
        anomaly_scores[anomalies],
        color="red",
        label="Anomalies (True Labels)",
        zorder=5,
    )

    # Add grid, labels, and title
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Anomaly Score", fontsize=12)
    ax.set_title("Anomaly Scores with True Labels", fontsize=14)

    # Add a legend
    ax.legend(loc="upper left", fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.savefig(filename)
