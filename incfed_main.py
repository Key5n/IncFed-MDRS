import os
from logging import getLogger

from experiments.utils.parser import args_parser
from experiments.utils.save_scores import save_scores
import numpy as np
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train_clients
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train_clients
from experiments.algorithms.IncFed.train import (
    evaluate_in_clients_incfed,
    train_in_clients_incfed,
)
from experiments.utils.logger import init_logger
from experiments.utils.smd import get_SMD_test_clients, get_SMD_train_clients

train = True
save = True


def incfed_main(
    dataset: str,
    result_dir: str,
    N_x: int = 200,
    leaking_rate: float = 1.0,
    rho: float = 0.95,
    input_scale: float = 0.001,
    beta=0.0001,
    trans_len=10,
    train: bool = True,
    save: bool = True,
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
    )

    save_scores(evaluation_results, result_dir)
    score = np.mean(
        [evaluation_result["PATE"] for evaluation_result in evaluation_results]
    )

    return score


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "ESN-SRE", "IncFed", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "IncFed.log"))

    incfed_main(
        dataset=dataset,
        result_dir=result_dir,
    )
