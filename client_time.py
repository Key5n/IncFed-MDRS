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

    N_x_list = [50]
    scores_list = []
    client_time_list = []
    server_time_list = []
    server_time_incfed_list = []

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
            server_time = [value[2] for value in values]
            server_time_incfed = [value[3] for value in values]

            client_time_save_path = os.path.join(
                result_dir_for_each_dataset, "client_time.csv"
            )
            server_time_save_path = os.path.join(
                result_dir_for_each_dataset, "server_time.csv"
            )
            server_time_incfed_save_path = os.path.join(
                result_dir_for_each_dataset, "server_time_incfed.csv"
            )
            np.savetxt(client_time_save_path, client_time)
            np.savetxt(server_time_save_path, server_time)
            np.savetxt(server_time_incfed_save_path, server_time_incfed)

        else:
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
            server_time_incfed = np.genfromtxt(
                os.path.join(
                    result_dir_for_each_dataset,
                    "server_time_incfed.csv",
                )
            )

        client_time_list.append(client_time)
        server_time_list.append(server_time)
        server_time_incfed_list.append(server_time_incfed)

    client_time_diagram_path = os.path.join(result_dir, "client_time.pdf")
    plot_line_graph(
        N_x_list,
        client_time_list,
        datasets,
        filepath=client_time_diagram_path,
        xlabel="The number of reservoir nodes",
        ylabel="Increase Time Ratio (%)",
    )

    server_time_diagram_path = os.path.join(result_dir, "server_time.pdf")
    plot_line_graph(
        N_x_list,
        server_time_list + server_time_incfed_list,
        datasets + ["SMD(IncFed MD-RS)", "SMAP(IncFed MD-RS)", "PSM(IncFed MD-RS)"],
        filepath=server_time_diagram_path,
        xlabel="The number of reservoir nodes",
        ylabel="Time (s)",
    )
