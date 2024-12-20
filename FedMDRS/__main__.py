import os
import pickle
from FedMDRS.utils.datasets import create_dataset
from FedMDRS.utils.optimize import optimize_federated_HP, optimize_isolated_HP

import json

from FedMDRS.utils.utils import (
    evaluate_in_client,
    evaluate_in_clients,
    train_in_client,
    train_in_clients,
)

train = True
isolated = False
federated = True
save = True
optimize = False

serverMachineDataset = create_dataset()

if federated:
    output_dir = os.path.join("result", "federated")
    os.makedirs(output_dir, exist_ok=True)
    if train:
        if optimize:
            leaking_rate, delta, rho, input_scale = optimize_federated_HP(
                serverMachineDataset, output_dir
            )

            with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                best_params = {}
                best_params["leaking_rate"] = leaking_rate
                best_params["delta"] = delta
                best_params["rho"] = rho
                best_params["input_scale"] = input_scale

                json.dump(best_params, f)
        elif os.path.isfile(os.path.join(output_dir, "best_params.json")):
            with open(os.path.join(output_dir, "best_params.json"), "r") as f:
                loaded_best_params = json.load(f)
                leaking_rate = loaded_best_params.leaking_rate
                delta = loaded_best_params.delta
                rho = loaded_best_params.rho
                input_scale = loaded_best_params.input_scale
        else:
            leaking_rate = 1.0
            delta = 0.0001
            rho = 0.95
            input_scale = 1.0

        models_dic = train_in_clients(
            serverMachineDataset,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
        )

        if save:
            print("global model is saved")
            with open(os.path.join(output_dir, "models.pickle"), "wb") as f:
                pickle.dump(models_dic, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(output_dir, "models.pickle"), "rb") as f:
            models_dic = pickle.load(f)

    auc_roc_avg, auc_pr_avg, vus_roc_avg, vus_pr_avg, pate_avg = evaluate_in_clients(
        models_dic, serverMachineDataset, output_dir
    )

    print(f"{auc_roc_avg = }")
    print(f"{auc_pr_avg = }")
    print(f"{vus_roc_avg = }")
    print(f"{vus_pr_avg = }")
    print(f"{pate_avg = }")

if isolated:
    print("isolated")
    output_dir = os.path.join("result", "isolated")
    os.makedirs(output_dir, exist_ok=True)

    for serverMachineData in serverMachineDataset:
        output_dir_client = os.path.join(output_dir, serverMachineData.data_name)
        os.makedirs(os.path.join(output_dir_client), exist_ok=True)

        if train:
            if optimize:
                leaking_rate, delta, rho, input_scale = optimize_isolated_HP(
                    serverMachineDataset, output_dir_client
                )

                with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                    best_params = {}
                    best_params["leaking_rate"] = leaking_rate
                    best_params["delta"] = delta
                    best_params["rho"] = rho
                    best_params["input_scale"] = input_scale

                    json.dump(best_params, f)
            elif os.path.isfile(os.path.join(output_dir_client, "best_params.json")):
                with open(
                    os.path.join(output_dir_client, "best_params.json"), "r"
                ) as f:
                    loaded_best_params = json.load(f)
                    leaking_rate = loaded_best_params.leaking_rate
                    delta = loaded_best_params.delta
                    rho = loaded_best_params.rho
                    input_scale = loaded_best_params.input_scale
            else:
                leaking_rate = 1.0
                delta = 0.0001
                rho = 0.95
                input_scale = 1.0

            model, _ = train_in_client(
                serverMachineData,
                leaking_rate=leaking_rate,
                delta=delta,
                rho=rho,
                input_scale=input_scale,
            )

            if save:
                with open(
                    os.path.join(output_dir_client, "model.pickle"),
                    "wb",
                ) as f:
                    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(
                os.path.join(output_dir_client, "model.pickle"),
                "rb",
            ) as f:
                model = pickle.load(f)

        auc_roc, auc_pr, vus_roc, vus_pr, pate = evaluate_in_client(
            model, serverMachineData, output_dir=output_dir
        )

        print(f"{auc_roc = }")
        print(f"{auc_pr = }")
        print(f"{vus_roc = }")
        print(f"{vus_pr = }")
        print(f"{pate = }")
