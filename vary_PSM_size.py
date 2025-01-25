from logging import getLogger
from experiments.utils.diagram.plot_line_graph import plot_line_graph
from fedmdrs_main import fedmdrs_main
import numpy as np
import os
from experiments.utils.logger import init_logger
from fedavg_tranad import fedavg_tranad
from tranad_main import tranad_main


if __name__ == "__main__":
    dataset = "PSM"
    result_dir = os.path.join("result", "data_proportion", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "tranad.log"))
    logger = getLogger(__name__)

    data_proportions = [0.01, 0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 0.875, 1]
    run = True

    if run:
        tranad_scores = [
            tranad_main(
                dataset=dataset,
                result_dir=os.path.join(result_dir, "tranad", str(data_proportion)),
                data_proportion=data_proportion,
            )
            for data_proportion in data_proportions
        ]
        fedavg_tranad_scores = [
            fedavg_tranad(
                dataset=dataset,
                result_dir=os.path.join(
                    result_dir, "fedavg_tranad", str(data_proportion)
                ),
                data_proportion=data_proportion,
            )
            for data_proportion in data_proportions
        ]
        proposed_scores = [
            fedmdrs_main(
                dataset=dataset,
                N_x=500,
                N_x_tilde=200,
                result_dir=os.path.join(result_dir, "proposed", str(data_proportion)),
                data_proportion=data_proportion,
            )
            for data_proportion in data_proportions
        ]
    else:
        tranad_scores = [
            np.mean(
                np.genfromtxt(
                    os.path.join(result_dir, "tranad", str(data_proportion), "pate.csv")
                )
            )
            for data_proportion in data_proportions
        ]
        fedavg_tranad_scores = [
            np.mean(
                np.genfromtxt(
                    os.path.join(
                        result_dir, "fedavg_tranad", str(data_proportion), "pate.csv"
                    )
                )
            )
            for data_proportion in data_proportions
        ]
        proposed_scores = [
            np.mean(
                np.genfromtxt(
                    os.path.join(
                        result_dir, "proposed", str(data_proportion), "pate.csv"
                    )
                )
            )
            for data_proportion in data_proportions
        ]

    figure_path = os.path.join(result_dir, "figure.pdf")
    Y_list = [tranad_scores, fedavg_tranad_scores, proposed_scores]
    label_list = ["TranAD", "FedAvg TranAD", "IncFed MDRS"]

    plot_line_graph(
        data_proportions,
        Y_list,
        label_list,
        filepath=figure_path,
        xlabel="Data Proportion",
    )
