import os
from experiments.utils.diagram.plot_line_graph import plot_line_graph
from incfedmdrswithpca_main import incfedmdrswithpca_main
import numpy as np
from experiments.utils.logger import init_logger

if __name__ == "__main__":
    datasets = ["SMD", "SMAP", "PSM"]
    result_dir = os.path.join("client_time", "mdrs", "pca")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))

    N_x_list = [50, 120, 250, 500, 750, 1000]
    scores_list = []
    client_time_list = []
    server_time_list = []

    for dataset in datasets:
        result_dir_for_each_dataset = os.path.join(result_dir, dataset)
        os.makedirs(result_dir_for_each_dataset, exist_ok=True)

        run = True
        if run:
            values = [
                incfedmdrswithpca_main(
                    dataset,
                    result_dir=os.path.join(result_dir_for_each_dataset, str(N_x)),
                    N_x=N_x,
                    N_x_tilde=None,
                    n_components=1,
                )
                for N_x in N_x_list
            ]
            client_time = [value[1] for value in values]

            client_time_save_path = os.path.join(
                result_dir_for_each_dataset, "client_time.csv"
            )
            np.savetxt(client_time_save_path, client_time)

        else:
            scores = [
                np.mean(
                    np.genfromtxt(
                        os.path.join(result_dir_for_each_dataset, str(N_x), "pate.csv")
                    )
                )
                for N_x in N_x_list
            ]
            client_time = np.genfromtxt(
                os.path.join(
                    result_dir_for_each_dataset,
                    "client_time.csv",
                )
            )

        client_time_list.append(client_time)

    client_time_diagram_path = os.path.join(result_dir, "client_time.pdf")
    plot_line_graph(
        N_x_list,
        client_time_list,
        datasets,
        filepath=client_time_diagram_path,
        xlabel="リザバーノード数",
        ylabel="増加時間割合（%）",
    )
