import os

from experiments.utils.parser import args_parser
import numpy as np
from FedMDRS.utils.fedavg_mdrs import train_in_clients_fedavg
from FedMDRS.utils.utils import evaluate_in_clients
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.logger import init_logger
from experiments.utils.smd import get_SMD_test_clients, get_SMD_train_clients
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train_clients
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train_clients

train = True
save = True


def fedavg_mdrs(
    dataset: "str",
    result_dir: str,
    N_x=200,
    leaking_rate=1.0,
    delta=0.0001,
    rho=0.95,
    input_scale=1.0,
):
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
        P_global = train_in_clients_fedavg(
            train_clients,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
        )

        if save:
            print("global model is saved")
            with open(os.path.join(result_dir, "P_global.npy"), "wb") as f:
                np.save(f, P_global)
    else:
        with open(os.path.join(result_dir, "P_global.npy"), "rb") as f:
            P_global = np.load(f)

    evaluation_results = evaluate_in_clients(test_clients, P_global, N_x, result_dir)

    get_final_scores(evaluation_results, result_dir)


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "mdrs", "fedavg", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "mdrs.log"))

    fedavg_mdrs(dataset=dataset, result_dir=result_dir)
