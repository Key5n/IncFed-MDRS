import os
import pickle
import optuna
import json
from utils import (
    create_dataset,
    train_in_clients,
    train_in_client,
    evaluate_in_clients,
    evaluate_in_client,
)

train = True
isolated = True
federated = True
save = True
optimize = True

serverMachineDataset = create_dataset()

if federated:
    output_dir = os.path.join("result", "federated")
    os.makedirs(output_dir, exist_ok=True)
    if train:
        if optimize:

            def federated_objective(trial):
                leaking_rate = trial.suggest_float(
                    "leaking_rate", 0.000001, 1, log=True
                )
                delta = trial.suggest_float("delta", 0.00001, 1, log=True)
                rho = trial.suggest_float("rho", 0, 2)
                input_scale = trial.suggest_float("input_scale", 0.0001, 1, log=True)
                model = train_in_clients(
                    serverMachineDataset,
                    leaking_rate=leaking_rate,
                    delta=delta,
                    rho=rho,
                    input_scale=input_scale,
                )
                pate_avg, VUS_PR_avg = evaluate_in_clients(
                    model, serverMachineDataset, output_dir=output_dir
                )
                print(f"{pate_avg = }, {VUS_PR_avg = }")

                return pate_avg

            study = optuna.create_study(direction="maximize")
            study.optimize(federated_objective, n_trials=50)

            leaking_rate = study.best_params["leaking_rate"]
            delta = study.best_params["delta"]
            rho = study.best_params["rho"]
            input_scale = study.best_params["input_scale"]

            with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                json.dump(study.best_params, f)
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

    evaluate_in_clients(models_dic, serverMachineDataset, output_dir)

if isolated:
    print("isolated")
    output_dir = os.path.join("result", "isolated")
    os.makedirs(output_dir, exist_ok=True)

    for serverMachineData in serverMachineDataset:
        if train:
            if optimize:

                def isolated_objective(trial):
                    leaking_rate = trial.suggest_float(
                        "leaking_rate", 0.00001, 1, log=True
                    )
                    delta = trial.suggest_float("delta", 0.00001, 1, log=True)
                    rho = trial.suggest_float("rho", 0, 1)
                    input_scale = trial.suggest_float(
                        "input_scale", 0.0001, 1, log=True
                    )
                    model, _ = train_in_client(
                        serverMachineData,
                        leaking_rate=leaking_rate,
                        delta=delta,
                        rho=rho,
                        input_scale=input_scale,
                    )
                    pate, _ = evaluate_in_client(
                        model, serverMachineData, output_dir=output_dir
                    )

                    return pate

                study = optuna.create_study(direction="maximize")
                study.optimize(isolated_objective, n_trials=50)

                leaking_rate = study.best_params["leaking_rate"]
                delta = study.best_params["delta"]
                rho = study.best_params["rho"]
                input_scale = study.best_params["input_scale"]

                with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                    json.dump(study.best_params, f)
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

            os.makedirs(
                os.path.join(output_dir, serverMachineData.data_name), exist_ok=True
            )
            model, _ = train_in_client(
                serverMachineData,
                leaking_rate=leaking_rate,
                delta=delta,
                rho=rho,
                input_scale=input_scale,
            )

            if save:
                with open(
                    os.path.join(
                        output_dir, serverMachineData.data_name, "model.pickle"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(
                os.path.join(output_dir, serverMachineData.data_name, "model.pickle"),
                "rb",
            ) as f:
                model = pickle.load(f)

        evaluate_in_client(model, serverMachineData, output_dir=output_dir)
