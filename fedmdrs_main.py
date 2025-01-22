import os
from logging import getLogger
import optuna
import joblib

import numpy as np
from FedMDRS.utils.utils import evaluate_in_clients, train_in_clients
from experiments.utils.parser import args_parser
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train_clients
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train_clients
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.logger import init_logger
from experiments.utils.smd import get_SMD_test_clients, get_SMD_train_clients


def fedmdrs_main(
    dataset: str,
    result_dir: str,
    N_x: int,
    leaking_rate: float = 1.0,
    delta: float = 0.0001,
    rho: float = 0.95,
    input_scale: float = 1.0,
    trans_len: int = 100,
    N_x_tilde=None,
    train: bool = True,
    save: bool = True,
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
        P_global = train_in_clients(
            train_clients,
            N_x=N_x,
            N_x_tilde=N_x_tilde,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
            trans_len=trans_len,
        )

        if save:
            print("saved global precision matrix")
            with open(os.path.join(result_dir, "P_global.npy"), "wb") as f:
                np.save(f, P_global)
    else:
        with open(os.path.join(result_dir, "P_global.npy"), "rb") as f:
            P_global = np.load(f)

    evaluation_results = evaluate_in_clients(
        test_clients,
        P_global,
        N_x,
        result_dir,
        N_x_tilde=N_x_tilde,
        leaking_rate=leaking_rate,
        delta=delta,
        rho=rho,
        input_scale=input_scale,
        trans_len=trans_len,
    )

    pate_avg = get_final_scores(evaluation_results, result_dir)

    return pate_avg


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "mdrs", "proposed", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))
    logger = getLogger(__name__)

    N_x = 500
    N_x_tilde = 200

    fedmdrs_main(
        dataset,
        result_dir=result_dir,
        N_x=N_x,
        N_x_tilde=N_x_tilde,
    )
