import os
from experiments.utils.diagram.plot_line_graph import plot_line_graph
from experiments.utils.diagram.subsampling import plot_subsampling
from incfedmdrswithpca_main import incfedmdrswithpca_main
import numpy as np
from experiments.utils.logger import init_logger

if __name__ == "__main__":
    datasets = ["SMD", "SMAP", "PSM"]
    result_dir = os.path.join("result", "mdrs", "pca")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))

    n_components_list = [
        1,
        5,
        10,
        100,
        200,
        400,
        600,
        800,
        1000,
    ]
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
                    result_dir=os.path.join(
                        result_dir_for_each_dataset, str(n_components)
                    ),
                    N_x=1000,
                    N_x_tilde=None,
                    n_components=n_components,
                )
                for n_components in n_components_list
            ]
            scores = [value[0] for value in values]
            client_time = [value[1] for value in values]
            server_time = [value[2] for value in values]

            client_time_save_path = os.path.join(result_dir_for_each_dataset, "client_time.csv")
            server_time_save_path = os.path.join(result_dir_for_each_dataset, "server_time.csv")
            np.savetxt(client_time_save_path, client_time)
            np.savetxt(server_time_save_path, server_time)

        else:
            scores = [
                np.mean(
                    np.genfromtxt(
                        os.path.join(
                            result_dir_for_each_dataset, str(n_components), "pate.csv"
                        )
                    )
                )
                for n_components in n_components_list
            ]
            client_time = np.genfromtxt(
                os.path.join(
                    result_dir_for_each_dataset,
                    "client_time.csv",
                )
            )

            server_time = np.genfromtxt(
                os.path.join(
                    result_dir_for_each_dataset,
                    "server_time.csv",
                )
            )

        scores_list.append(scores)
        client_time_list.append(client_time)
        server_time_list.append(server_time)
    diagram_path = os.path.join(result_dir, "pca-result.pdf")
    plot_line_graph(
        n_components_list,
        scores_list,
        xlabel="the number of components",
        label_list=datasets,
        filepath=diagram_path,
    )

    client_time_diagram_path = os.path.join(result_dir, "client_time.pdf")
    plot_line_graph(
        n_components_list,
        client_time_list,
        datasets,
        filepath=client_time_diagram_path,
        xlabel="the number of components",
        ylabel="Time (s)"
    )

    server_time_diagram_path = os.path.join(result_dir, "server_time.pdf")
    plot_line_graph(
        n_components_list,
        server_time_list,
        datasets,
        filepath=server_time_diagram_path,
        xlabel="the number of components",
        ylabel="Time (s)"
    )
