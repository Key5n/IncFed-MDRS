import os
from typing import Dict
from tqdm import tqdm
from tqdm.contrib import tenumerate
from experiments.evaluation.metrics import get_metrics
import numpy as np
from numpy.typing import NDArray
from experiments.algorithms.IncFed.layers import ESN
from experiments.utils.diagram.plot import plot


def train_in_clients_incfed(
    train_data_list: NDArray,
    N_x: int,
    input_scale: float,
    leaking_rate: float,
    rho: float,
    beta: float,
    trans_len: int = 10,
) -> tuple[NDArray, NDArray]:
    N_y = train_data_list[0].shape[1]
    A: NDArray = np.zeros((N_y, N_x))
    B: NDArray = np.zeros((N_x, N_x))

    for train_data in tqdm(train_data_list):
        Ac, Bc = train_in_client_incfed(
            train_data,
            N_x,
            input_scale=input_scale,
            leaking_rate=leaking_rate,
            rho=rho,
            beta=beta,
            trans_len=trans_len,
        )
        A += Ac
        B += Bc

    return A, B


def train_in_client_incfed(
    train_data: NDArray,
    N_x: int,
    input_scale: float,
    leaking_rate: float,
    rho: float,
    beta: float,
    trans_len: int,
) -> tuple[NDArray, NDArray]:
    N_u = train_data.shape[1]
    N_y = N_u

    model = ESN(
        N_u,
        N_y,
        N_x,
        input_scale=input_scale,
        leaking_rate=leaking_rate,
        beta=beta,
        rho=rho,
    )
    target_data = train_data
    A_c, B_c = model.train(train_data, target_data, trans_len)

    return A_c, B_c


def evaluate_in_clients_incfed(
    test_clients: list[tuple[NDArray, NDArray]],
    W_out: NDArray,
    N_x: int,
    input_scale: float,
    leaking_rate: float,
    rho: float,
    beta: float,
    trans_len: int,
    result_dir: str,
):
    evaluation_results: list[Dict] = []
    for i, (test_data, test_label) in tenumerate(test_clients):
        evaluation_result = evaluate_in_client_incfed(
            test_data,
            test_label,
            W_out,
            N_x,
            input_scale=input_scale,
            leaking_rate=leaking_rate,
            beta=beta,
            rho=rho,
            trans_len=trans_len,
            filename=os.path.join(result_dir, f"{i}.pdf"),
        )

        evaluation_results.append(evaluation_result)

    return evaluation_results


def evaluate_in_client_incfed(
    test_data: NDArray,
    test_label: NDArray,
    W_out: NDArray,
    N_x: int,
    input_scale: float,
    leaking_rate: float,
    rho: float,
    beta: float,
    trans_len: int,
    filename: str,
):
    N_u = test_data.shape[1]
    N_y = N_u
    model = ESN(
        N_u,
        N_y,
        N_x,
        input_scale=input_scale,
        leaking_rate=leaking_rate,
        beta=beta,
        rho=rho,
    )
    target_data = test_data
    scores = model.predict(test_data, target_data, W_out, trans_len=trans_len)

    plot(scores, test_label, filename)
    evaluation_result = get_metrics(scores, test_label)

    return evaluation_result
