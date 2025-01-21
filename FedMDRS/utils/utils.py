import os
from typing import Dict

from tqdm import tqdm
from tqdm.contrib import tenumerate
from experiments.utils.diagram.plot import plot
import numpy as np
from numpy.typing import NDArray
from FedMDRS.mdrs import MDRS
from experiments.evaluation.metrics import get_metrics


def train_in_clients(
    train_data_list: list[NDArray],
    N_x: int,
    leaking_rate,
    rho,
    delta,
    input_scale: float,
    N_x_tilde: int | None = None,
) -> NDArray:

    if N_x_tilde is None:
        N_x_tilde = N_x

    covariance_matrix = delta * np.identity(N_x_tilde)
    for train_data in tqdm(train_data_list):
        local_updates = train_in_client(
            train_data,
            N_x,
            N_x_tilde=N_x_tilde,
            leaking_rate=leaking_rate,
            rho=rho,
            delta=delta,
            input_scale=input_scale,
        )

        covariance_matrix += local_updates

    P_global = np.linalg.inv(covariance_matrix)

    return P_global


def train_in_client(
    train_data: NDArray,
    N_x: int,
    leaking_rate,
    rho,
    delta,
    input_scale: float,
    N_x_tilde: int | None = None,
) -> NDArray:
    N_u = train_data.shape[1]
    model = MDRS(
        N_u,
        N_x,
        N_x_tilde=N_x_tilde,
        leaking_rate=leaking_rate,
        delta=delta,
        rho=rho,
        input_scale=input_scale,
    )
    local_updates = model.train(train_data)

    return local_updates


def evaluate_in_clients(
    test_data_list: list[tuple[NDArray, NDArray]],
    P_global: NDArray,
    N_x: int,
    result_dir: str,
    leaking_rate: float,
    rho: float,
    delta: float,
    input_scale: float,
    N_x_tilde: int | None = None,
) -> list[Dict]:
    evaluation_results: list[Dict] = []
    for i, (test_data, test_label) in tenumerate(test_data_list):
        evaluation_result = evaluate_in_client(
            test_data,
            test_label,
            P_global=P_global,
            N_x=N_x,
            filename=os.path.join(result_dir, f"{i}.pdf"),
            N_x_tilde=N_x_tilde,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
        )

        evaluation_results.append(evaluation_result)

    return evaluation_results


def evaluate_in_client(
    test_data: NDArray,
    test_label: NDArray,
    P_global: NDArray,
    N_x: int,
    filename: str,
    leaking_rate: float,
    rho: float,
    delta: float,
    input_scale: float,
    N_x_tilde: int | None = None,
) -> Dict:
    N_u = test_data.shape[1]
    model = MDRS(
        N_u,
        N_x,
        N_x_tilde=N_x_tilde,
        precision_matrix=P_global,
        leaking_rate=leaking_rate,
        delta=delta,
        rho=rho,
        input_scale=input_scale,
    )
    scores = model.adapt(test_data)

    plot(scores, test_label, filename)
    evaluation_result = get_metrics(scores, test_label)

    return evaluation_result
