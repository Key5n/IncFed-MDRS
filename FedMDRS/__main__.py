import os

import numpy as np
from utils.datasets import get_dataset

from utils.utils import (
    evaluate,
    train_in_clients,
)

train = True
federated = True
save = True

entities, X_test, y_test = get_dataset("SMD")

if federated:
    output_dir = os.path.join("result", "federated")
    os.makedirs(output_dir, exist_ok=True)

    leaking_rate = 1.0
    delta = 0.0001
    rho = 0.95
    input_scale = 1.0

    if train:
        P_global = train_in_clients(
            entities,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
        )

        if save:
            print("global model is saved")
            with open(os.path.join(output_dir, "P_global.npy"), "wb") as f:
                np.save(f, P_global)
    else:
        with open(os.path.join(output_dir, "P_global.npy"), "rb") as f:
            P_global = np.load(f)

    auc_roc, auc_pr, vus_roc, vus_pr, pate = evaluate(
        X_test,
        y_test,
        P_global,
        leaking_rate=leaking_rate,
        delta=delta,
        input_scale=input_scale,
        rho=rho,
    )

    print(f"{auc_roc = }")
    print(f"{auc_pr = }")
    print(f"{vus_roc = }")
    print(f"{vus_pr = }")
    print(f"{pate = }")
