import optuna
from .utils import (
    evaluate_in_client,
    evaluate_in_clients,
    train_in_client,
    train_in_clients,
)


def optimize_federated_HP(
    datasets, output_dir: str
) -> tuple[float, float, float, float]:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        get_federated_objective_function(datasets, output_dir),
        n_trials=20,
    )

    best_leaking_rate = study.best_params["leaking_rate"]
    best_delta = study.best_params["delta"]
    best_rho = study.best_params["rho"]
    best_input_scale = study.best_params["input_scale"]

    return best_leaking_rate, best_delta, best_rho, best_input_scale


def get_federated_objective_function(datasets, output_dir: str):
    def federated_objective(trial) -> float:
        leaking_rate = trial.suggest_float("leaking_rate", 0.0001, 1, log=True)
        delta = trial.suggest_float("delta", 0.0001, 1, log=True)
        rho = trial.suggest_float("rho", 0, 2)
        input_scale = trial.suggest_float("input_scale", 0.0001, 1, log=True)
        model = train_in_clients(
            datasets,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
        )
        auc_roc_avg, auc_pr_avg, vus_roc_avg, vus_pr_avg, pate_avg = (
            evaluate_in_clients(model, datasets, output_dir=output_dir)
        )

        print(f"{auc_roc_avg = }")
        print(f"{auc_pr_avg = }")
        print(f"{vus_roc_avg = }")
        print(f"{vus_pr_avg = }")
        print(f"{pate_avg = }")

        return pate_avg

    return federated_objective


def optimize_isolated_HP(
    datasets, output_dir: str
) -> tuple[float, float, float, float]:
    study = optuna.create_study(direction="maximize")
    study.optimize(get_isolated_objective_function(datasets, output_dir), n_trials=20)

    best_leaking_rate = study.best_params["leaking_rate"]
    best_delta = study.best_params["delta"]
    best_rho = study.best_params["rho"]
    best_input_scale = study.best_params["input_scale"]

    return best_leaking_rate, best_delta, best_rho, best_input_scale


def get_isolated_objective_function(datasets, output_dir: str):
    def isolated_objective(trial):
        leaking_rate = trial.suggest_float("leaking_rate", 0.0001, 1, log=True)
        delta = trial.suggest_float("delta", 0.0001, 1, log=True)
        rho = trial.suggest_float("rho", 0, 1)
        input_scale = trial.suggest_float("input_scale", 0.0001, 1, log=True)
        model, _ = train_in_client(
            datasets,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
        )
        auc_roc, auc_pr, vus_roc, vus_pr, pate = evaluate_in_client(
            model, datasets, output_dir=output_dir
        )

        print(f"{auc_roc = }")
        print(f"{auc_pr = }")
        print(f"{vus_roc = }")
        print(f"{vus_pr = }")
        print(f"{pate = }")

        return pate

    return isolated_objective
