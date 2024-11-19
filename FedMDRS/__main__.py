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
individually = True
federated = True
save = True
optimize = True

serverMachineDataset = create_dataset()

if federated:
    if train:
        if optimize:
            def federated_objective(trial):
                leaking_rate = trial.suggest_float("learking_rate", 0, 1)
                delta = trial.suggest_float("delta", 0, 1)
                rho = trial.suggest_float("rho", 0, 1)
                model, P_global = train_in_clients(serverMachineDataset, leaking_rate=leaking_rate, delta=delta, rho=rho)
                pr_curve_auc_average = evaluate_in_clients(P_global, model, serverMachineDataset)

                return pr_curve_auc_average

            study = optuna.create_study(direction="maximize")
            study.optimize(federated_objective, n_trials=100)

            leaking_rate = study.best_params["leaking_rate"]
            delta = study.best_params["delta"]
            rho = study.best_params["rho"]

            with open("result/best_params.json", "w") as f:
                json.dump(study.best_params, f)
        elif os.path.isfile("result/best_params.json"):
            with open(f"result/best_params.json", "r") as f:
                loaded_best_params = json.load(f)
                leaking_rate = loaded_best_params.leaking_rate
                delta = loaded_best_params.delta
                rho = loaded_best_params.rho
        else:
            leaking_rate = 1.0
            delta = 0.0001
            rho = 0.95

        models_dic, P_global = train_in_clients(serverMachineDataset, leaking_rate=leaking_rate, delta=delta, rho=rho)

        if save:
            print("model and precision matrix are saved")
            with open("models.pickle", "wb") as f:
                pickle.dump(models_dic, f, pickle.HIGHEST_PROTOCOL)
            np.savetxt("P_global.txt", P_global, delimiter=",")
    else:
        P_global = np.genfromtxt("P_global.txt", dtype=np.float64, delimiter=",")
        with open("models.pickle", "rb") as f:
            models_dic = pickle.load(f)

    evaluate_in_clients(P_global, models_dic, serverMachineDataset)

if individually:
    print("individually")
    output_dir = "individual"

    for serverMachineData in serverMachineDataset:
        if train:
            if optimize:
                def individual_objective(trial):
                    leaking_rate = trial.suggest_float("leaking_rate", 0, 1)
                    delta = trial.suggest_float("delta", 0, 1)
                    rho = trial.suggest_float("rho", 0, 1)
                    model, _ = train_in_client(serverMachineData, leaking_rate=leaking_rate, delta=delta, rho=rho)
                    pr_curve_auc = evaluate_in_client(model, serverMachineData, output_dir="individual")

                    return pr_curve_auc

                study = optuna.create_study(direction="maximize")
                study.optimize(individual_objective, n_trials=100)

                leaking_rate = study.best_params["leaking_rate"]
                delta = study.best_params["delta"]
                rho = study.best_params["rho"]

                with open(f"{output_dir}/best_params.json", "w") as f:
                    json.dump(study.best_params, f)
            elif os.path.isfile(f"{output_dir}/best_params.json"):
                with open(f"{output_dir}/best_params.json", "r") as f:
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
                print("individual model is saved")
                with open(f"{output_dir}/{serverMachineData.data_name}/model.pickle", "wb") as f:
                    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f"{output_dir}/{serverMachineData.data_name}/model.pickle", "rb") as f:
                model = pickle.load(f)

        evaluate_in_client(model, serverMachineData, output_dir=output_dir)
