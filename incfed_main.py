import os
from logging import getLogger
import json

from experiments.utils.parser import args_parser
import optuna
import numpy as np
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train_clients
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train_clients
from experiments.algorithms.IncFed.train import (
    evaluate_in_clients_incfed,
    train_in_clients_incfed,
)
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.logger import init_logger
from experiments.utils.smd import get_SMD_test_clients, get_SMD_train_clients

train = True
save = True


def incfed_main(
    dataset: str = "SMAP",
    N_x: int = 200,
    leaking_rate: float = 1.0,
    rho: float = 0.95,
    input_scale: float = 1.0,
    beta=0.0001,
    trans_len=10,
    train: bool = True,
    save: bool = True,
    result_dir: str = "result",
):
    config = locals()
    logger = getLogger(__name__)
    logger.info(config)

    if dataset == "SMD":
        train_clients = get_SMD_train_clients()
        test_clients = get_SMD_test_clients()
    elif dataset == "SMAP":
        train_clients = get_SMAP_train_clients()
        test_clients = get_SMAP_test_clients()
    else:
        num_clients = 24
        train_clients = get_PSM_train_clients(num_clients)
        test_clients = get_PSM_test_clients()

    if train:
        A, B = train_in_clients_incfed(
            train_clients,
            N_x,
            input_scale=input_scale,
            leaking_rate=leaking_rate,
            rho=rho,
            trans_len=trans_len,
            beta=beta,
        )

        W_out = np.dot(A, np.linalg.inv(B))
        if save:
            with open(os.path.join(result_dir, "W_out.npy"), "wb") as f:
                np.save(f, W_out)

            print("saved W_out")
    else:
        with open(os.path.join(result_dir, "W_out.npy"), "rb") as f:
            W_out = np.load(f)

    evaluation_results = evaluate_in_clients_incfed(
        test_clients,
        W_out,
        N_x,
        input_scale=input_scale,
        leaking_rate=leaking_rate,
        rho=rho,
        beta=beta,
        trans_len=trans_len,
        result_dir=result_dir,
    )

    pate_avg = get_final_scores(evaluation_results, result_dir)

    return pate_avg


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "ESN-SRE", "IncFed", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "IncFed.log"))

    leaking_rate = 1.0
    beta = 0.0001
    rho = 0.95
    input_scale = 1.0

    incfed_main(
        dataset=dataset,
        leaking_rate=leaking_rate,
        rho=rho,
        input_scale=input_scale,
        beta=beta,
        save=False,
        result_dir=result_dir,
    )
