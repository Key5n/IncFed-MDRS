import os
import pickle
import numpy as np
import optuna
import json
from utils import create_dataset, train_in_clients, train_in_client, evaluate_in_clients, evaluate_in_client

dirname = "ServerMachineDataset"
train_data_dir_path = os.path.join(dirname, "train")
test_data_dir_path = os.path.join(dirname, "test")
test_label_dir_path = os.path.join(dirname, "test_label")

train = True
isolated = True
federated = True
save = True
optimize = True

serverMachineDataset = create_dataset()

if federated:
    output_dir = os.path.join("result", "federated")
    if train:
        if optimize:
            def federated_objective(trial):
                leaking_rate = trial.suggest_float("leaking_rate", 0, 1)
                delta = trial.suggest_float("delta", 0, 1)
                rho = trial.suggest_float("rho", 0, 1)
                model = train_in_clients(serverMachineDataset, leaking_rate=leaking_rate, delta=delta, rho=rho)
                pr_curve_auc_average, _ = evaluate_in_clients(model, serverMachineDataset)

                return pr_curve_auc_average

            study = optuna.create_study(direction="maximize")
            study.optimize(federated_objective, n_trials=100)

            leaking_rate = study.best_params["leaking_rate"]
            delta = study.best_params["delta"]
            rho = study.best_params["rho"]

            with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                json.dump(study.best_params, f)
        elif os.path.isfile(os.path.join(output_dir, "best_params.json")):
            with open(os.path.join(output_dir, "best_params.json"), "r") as f:
                loaded_best_params = json.load(f)
                leaking_rate = loaded_best_params.leaking_rate
                delta = loaded_best_params.delta
                rho = loaded_best_params.rho
        else:
            leaking_rate = 1.0
            delta = 0.0001
            rho = 0.95

        models_dic = train_in_clients(serverMachineDataset, leaking_rate=leaking_rate, delta=delta, rho=rho)

        if save:
            print("global model is saved")
            with open("models.pickle", "wb") as f:
                pickle.dump(models_dic, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open("models.pickle", "rb") as f:
            models_dic = pickle.load(f)

    evaluate_in_clients(models_dic, serverMachineDataset)

if isolated:
    print("isolated")
    output_dir = os.path.join("result", "isolated")

    for serverMachineData in serverMachineDataset:
        if train:
            if optimize:
                def isolated_objective(trial):
                    leaking_rate = trial.suggest_float("leaking_rate", 0, 1)
                    delta = trial.suggest_float("delta", 0, 1)
                    rho = trial.suggest_float("rho", 0, 1)
                    model, _ = train_in_client(serverMachineData, leaking_rate=leaking_rate, delta=delta, rho=rho)
                    pr_curve_auc = evaluate_in_client(model, serverMachineData, output_dir=output_dir)

                    return pr_curve_auc

                study = optuna.create_study(direction="maximize")
                study.optimize(isolated_objective, n_trials=100)

                leaking_rate = study.best_params["leaking_rate"]
                delta = study.best_params["delta"]
                rho = study.best_params["rho"]

                with open(os.path.join(output_dir, "best_params.json"), "w") as f:
                    json.dump(study.best_params, f)
            elif os.path.isfile(os.path.join(output_dir, "best_params.json")):
                with open(os.path.join(output_dir, "best_params.json"), "r") as f:
                    loaded_best_params = json.load(f)
                    leaking_rate = loaded_best_params.leaking_rate
                    delta = loaded_best_params.delta
                    rho = loaded_best_params.rho
            else:
                leaking_rate = 1.0
                delta = 0.0001
                rho = 0.95

            model, _ = train_in_client(serverMachineData, leaking_rate=leaking_rate, delta=delta, rho=rho)

            if save:
                print("isolated model is saved")
                with open(os.path.join(output_dir, serverMachineData.data_name, "model.pickle"), "wb") as f:
                    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(output_dir, serverMachineData.data_name, "model.pickle"), "rb") as f:
                model = pickle.load(f)

        evaluate_in_client(model, serverMachineData, output_dir=output_dir)
