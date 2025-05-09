from logging import getLogger
from experiments.utils.diagram.plot_line_graph import plot_line_graph
from fedavg_lstmae import fedavg_lstmae
from fedavg_mdrs import fedavg_mdrs
from incfedmdrs_main import incfedmdrs_main
from incfedesn_main import incfed_main
import numpy as np
import os
from experiments.utils.logger import init_logger
from fedavg_tranad import fedavg_tranad


if __name__ == "__main__":
    dataset = "PSM"
    result_dir = os.path.join("result", "client_size", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "tranad.log"))
    logger = getLogger(__name__)

    run = True
    client_sizes = [1, 2, 4, 8, 16, 24, 32, 48]

    if run:
        fedavg_tranad_scores = [
            fedavg_tranad(
                dataset=dataset,
                result_dir=os.path.join(result_dir, "tranad", str(num_clients)),
                num_clients=num_clients,
            )
            for num_clients in client_sizes
        ]
        # fedavg_lstmae_scores = [
        #     fedavg_lstmae(
        #         dataset=dataset,
        #         result_dir=os.path.join(result_dir, "lstmae", str(num_clients)),
        #         num_clients=num_clients,
        #     )
        #     for num_clients in client_sizes
        # ]
        incfed_scores = [
            incfed_main(
                dataset=dataset,
                result_dir=os.path.join(result_dir, "incfed", str(num_clients)),
                num_clients=num_clients,
            )
            for num_clients in client_sizes
        ]
        fedavg_mdrs_scores = [
            fedavg_mdrs(
                dataset=dataset,
                result_dir=os.path.join(result_dir, "fedavg_mdrs", str(num_clients)),
                num_clients=num_clients,
            )
            for num_clients in client_sizes
        ]
        proposed_scores = [
            incfedmdrs_main(
                dataset=dataset,
                N_x=500,
                N_x_tilde=200,
                result_dir=os.path.join(result_dir, "proposed", str(num_clients)),
                num_clients=num_clients,
            )
            for num_clients in client_sizes
        ]
    else:
        fedavg_tranad_scores = [
            np.mean(
                np.genfromtxt(
                    os.path.join(result_dir, "tranad", str(num_clients), "pate.csv")
                )
            )
            for num_clients in client_sizes
        ]
        # fedavg_lstmae_scores = [
        #     np.mean(
        #         np.genfromtxt(
        #             os.path.join(result_dir, "lstmae", str(num_clients), "pate.csv")
        #         )
        #     )
        #     for num_clients in client_sizes
        # ]
        incfed_scores = [
            np.mean(
                np.genfromtxt(
                    os.path.join(result_dir, "incfed", str(num_clients), "pate.csv")
                )
            )
            for num_clients in client_sizes
        ]
        fedavg_mdrs_scores = [
            np.mean(
                np.genfromtxt(
                    os.path.join(
                        result_dir, "fedavg_mdrs", str(num_clients), "pate.csv"
                    )
                )
            )
            for num_clients in client_sizes
        ]
        proposed_scores = [
            np.mean(
                np.genfromtxt(
                    os.path.join(result_dir, "proposed", str(num_clients), "pate.csv")
                )
            )
            for num_clients in client_sizes
        ]

    figure_path = os.path.join(result_dir, "vary_client_size.pdf")
    Y_list = [fedavg_tranad_scores, incfed_scores, fedavg_mdrs_scores, proposed_scores]
    label_list = ["FedAvg TranAD", "IncFed ESN-SRE", "FedAvg MD-RS", "IncFed MD-RS"]

    plot_line_graph(
        client_sizes, Y_list, label_list, filepath=figure_path, xlabel="Client Size"
    )
