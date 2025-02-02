import os
from experiments.utils.diagram.subsampling import plot_subsampling
import numpy as np
from experiments.utils.logger import init_logger
from incfedmdrs_main import incfedmdrs_main


if __name__ == "__main__":
    datasets = ["SMD", "SMAP", "PSM"]
    result_dir = os.path.join("result", "mdrs", "subsampling")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))

    subsampling_sizes = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    scores_list = []
    stds_list = []
    for dataset in datasets:
        result_dir_for_each_dataset = os.path.join(result_dir, dataset)
        os.makedirs(result_dir_for_each_dataset, exist_ok=True)

        N_x = 500
        run = False
        if run:
            scores = [
                incfedmdrs_main(
                    dataset,
                    result_dir=os.path.join(
                        result_dir_for_each_dataset, str(N_x_tilde)
                    ),
                    N_x=N_x,
                    N_x_tilde=N_x_tilde,
                )
                for N_x_tilde in subsampling_sizes
            ]
        else:
            scores = [
                np.mean(
                    np.genfromtxt(
                        os.path.join(
                            result_dir_for_each_dataset, str(N_x_tilde), "pate.csv"
                        )
                    )
                )
                for N_x_tilde in subsampling_sizes
            ]

        if dataset != "PSM":
            pate_stds = [
                np.std(
                    np.genfromtxt(
                        os.path.join(
                            result_dir_for_each_dataset, str(N_x_tilde), "pate.csv"
                        )
                    )
                )
                for N_x_tilde in subsampling_sizes
            ]
        else:
            pate_stds = np.zeros(len(subsampling_sizes))

        scores_list.append(scores)
        stds_list.append(pate_stds)

    diagram_path = os.path.join(result_dir, "subsampling-result.pdf")
    plot_subsampling(
        subsampling_sizes,
        scores_list,
        labels=datasets,
        filename=diagram_path,
    )
