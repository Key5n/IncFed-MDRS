import os
from logging import getLogger

from experiments.utils.parser import args_parser
from experiments.utils.save_scores import save_scores
import numpy as np
from FedMDRS.utils.fedavg_mdrs import train_in_clients_fedavg
from FedMDRS.utils.utils import evaluate_in_clients
from experiments.utils.logger import init_logger
from experiments.utils.smd import get_SMD_test_clients, get_SMD_train_clients
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train_clients
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train_clients

train = True
save = True


def fedavg_mdrs(
    dataset: str,
    result_dir: str,
    N_x: int = 500,
    leaking_rate=1.0,
    delta=0.0001,
    rho=0.95,
    input_scale=0.001,
    trans_len=10,
    N_x_tilde: int | None = 200,
    num_clients: int = 24,
):
    config = locals()
    logger = getLogger(__name__)
    logger.info(config)
    os.makedirs(result_dir, exist_ok=True)

    if dataset == "SMD":
        train_clients = get_SMD_train_clients()
        test_clients = get_SMD_test_clients()
    elif dataset == "SMAP":
        train_clients = get_SMAP_train_clients()
        test_clients = get_SMAP_test_clients()
    else:
        train_clients = get_PSM_train_clients(num_clients, required_length=trans_len)
        test_clients = get_PSM_test_clients()

    if train:
        P_global = train_in_clients_fedavg(
            train_clients,
            N_x,
            N_x_tilde=N_x_tilde,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
            trans_len=trans_len,
        )

        if save:
            print("global model is saved")
            with open(os.path.join(result_dir, "P_global.npy"), "wb") as f:
                np.save(f, P_global)
    else:
        with open(os.path.join(result_dir, "P_global.npy"), "rb") as f:
            P_global = np.load(f)

    evaluation_results = evaluate_in_clients(
        test_clients,
        P_global,
        N_x,
        N_x_tilde=N_x_tilde,
        leaking_rate=leaking_rate,
        delta=delta,
        rho=rho,
        input_scale=input_scale,
        trans_len=trans_len,
    )

    save_scores(evaluation_results, result_dir)
    score = np.mean(
        [evaluation_result["PATE"] for evaluation_result in evaluation_results]
    )

    return score


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "mdrs", "fedavg", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))

    fedavg_mdrs(dataset=dataset, result_dir=result_dir)
