import os
from logging import getLogger

import numpy as np
from experiments.algorithms.IncFed.train import (
    evaluate_in_clients_incfed,
    train_in_clients_incfed,
)
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.logger import init_logger
from experiments.utils.smd import get_SMD_test_clients, get_SMD_train_clients

train = True
save = True


if __name__ == "__main__":
    result_dir = os.path.join("result", "ESN-SRE", "IncFed")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "IncFed.log"))
    logger = getLogger(__name__)

    train_clients = get_SMD_train_clients()
    test_clients = get_SMD_test_clients()

    leaking_rate = 1.0
    rho = 0.95
    input_scale = 1.0
    trans_len = 10
    beta = 0.0001
    N_x = 200

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

    get_final_scores(evaluation_results, result_dir)
