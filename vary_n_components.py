import os
from experiments.utils.diagram.subsampling import plot_subsampling
from incfedmdrswithpca_main import incfedmdrswithpca_main
import numpy as np
from experiments.utils.logger import init_logger

if __name__ == "__main__":
    datasets = ["SMD", "SMAP", "PSM"]
    result_dir = os.path.join("result", "mdrs", "pca")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))

    n_components_list = [1, 5, 10, 25, 50, 100, 150, 200]
    scores_list = []

    for dataset in datasets:
        result_dir_for_each_dataset = os.path.join(result_dir, dataset)
        os.makedirs(result_dir_for_each_dataset, exist_ok=True)

        run = True
        if run:
            scores = [
                incfedmdrswithpca_main(
                    dataset, result_dir=result_dir, n_components=n_components
                )
                for n_components in n_components_list
            ]
            scores_list.append(scores)

    diagram_path = os.path.join(result_dir, "pca-result.pdf")
    plot_subsampling(
        n_components_list,
        scores_list,
        labels=datasets,
        filename=diagram_path,
    )
