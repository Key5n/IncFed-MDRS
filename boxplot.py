import os

import numpy as np
import matplotlib.pyplot as plt


def boxplot(
    X, tick_labels: list[str], colors: list[tuple], x_label: str, result_path: str
):
    _, ax = plt.subplots()
    X = np.flip(X).tolist()
    tick_labels = np.flip(tick_labels).tolist()

    bplot = ax.boxplot(
        X,
        tick_labels=tick_labels,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        orientation="horizontal",
    )

    for median in bplot["medians"]:
        median.set_color("black")

    for median in bplot["means"]:
        median.set_color("black")

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel(x_label.upper())
    plt.tight_layout()
    plt.savefig(result_path)

    plt.clf()
    plt.close()


result_dir = os.path.join(os.getcwd(), "result")

datasets = ["SMD", "SMAP"]
metrics = ["auc_roc", "auc_pr", "vus_pr", "pate"]


tick_labels = [
    "TranAD",
    "LSTM-AE",
    "FedAvg TranAD",
    "FedAvg LSTM-AE",
    "IncFed ESN-SRE",
    "FedAvg MD-RS",
    "IncFed MD-RS",
]

colors = [
    (216 / 255, 85 / 255, 255 / 255, 1 - i / len(tick_labels))
    for i in range(len(tick_labels))
]

for dataset in datasets:
    for metric in metrics:
        result_path = os.path.join(result_dir, f"{dataset}-{metric}.pdf")

        X = [
            np.genfromtxt(
                os.path.join(
                    os.getcwd(), f"result/tranad/centralized/{dataset}/{metric}.csv"
                )
            ),
            np.genfromtxt(
                os.path.join(
                    os.getcwd(), f"result/lstmae/centralized/{dataset}/{metric}.csv"
                )
            ),
            np.genfromtxt(
                os.path.join(
                    os.getcwd(), f"result/tranad/fedavg/{dataset}/{metric}.csv"
                )
            ),
            np.genfromtxt(
                os.path.join(
                    os.getcwd(), f"result/lstmae/fedavg/{dataset}/{metric}.csv"
                )
            ),
            np.genfromtxt(
                os.path.join(
                    os.getcwd(), f"result/ESN-SRE/IncFed/{dataset}/{metric}.csv"
                )
            ),
            np.genfromtxt(
                os.path.join(os.getcwd(), f"result/mdrs/fedavg/{dataset}/{metric}.csv")
            ),
            np.genfromtxt(
                os.path.join(
                    os.getcwd(), f"result/mdrs/proposed/{dataset}/{metric}.csv"
                )
            ),
        ]

        boxplot(
            X,
            tick_labels=tick_labels,
            colors=colors,
            x_label=metric,
            result_path=result_path,
        )
