from logging import getLogger
import os
from experiments.utils.diagram.subsampling import plot_subsampling
import numpy as np
from experiments.utils.logger import init_logger
from experiments.utils.parser import args_parser
from fedmdrs_main import fedmdrs_main


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "mdrs", "subsampling", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))
    logger = getLogger(__name__)

    N_x = 500

    subsampling_sizes = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    run = True

    if run:
        scores = [
            fedmdrs_main(
                dataset,
                result_dir=os.path.join(result_dir, str(N_x_tilde)),
                N_x=N_x,
                N_x_tilde=N_x_tilde,
            )
            for N_x_tilde in subsampling_sizes
        ]
    else:
        scores = [
            np.mean(np.genfromtxt(os.path.join(result_dir, str(N_x_tilde), "PATE.csv")))
            for N_x_tilde in subsampling_sizes
        ]

    pate_stds = [
        np.std(np.genfromtxt(os.path.join(result_dir, str(N_x_tilde), "PATE.csv")))
        for N_x_tilde in subsampling_sizes
    ]

    diagram_path = os.path.join(result_dir, "diagram.pdf")
    plot_subsampling(subsampling_sizes, scores, pate_stds, filename=diagram_path)
