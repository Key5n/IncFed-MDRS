from logging import getLogger
from typing import Dict
from tqdm import tqdm
from experiments.evaluation.metrics import get_metrics
import numpy as np
from numpy.typing import NDArray
from experiments.algorithms.IncFed.layers import ESN


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
):
    logger = getLogger(__name__)
    evaluation_results: list[Dict] = []

    for test_data, test_label in tqdm(test_clients):
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
        )

        evaluation_results.append(evaluation_result)

    auc_roc_scores = [
        evaluation_result["AUC-ROC"] for evaluation_result in evaluation_results
    ]
    auc_pr_scores = [
        evaluation_result["AUC-PR"] for evaluation_result in evaluation_results
    ]
    vus_roc_scores = [
        evaluation_result["VUS-ROC"] for evaluation_result in evaluation_results
    ]
    vus_pr_scores = [
        evaluation_result["VUS-PR"] for evaluation_result in evaluation_results
    ]
    pate_scores = [
        evaluation_result["PATE"] for evaluation_result in evaluation_results
    ]

    auc_roc_avg = np.mean(auc_roc_scores, dtype=float)
    auc_pr_avg = np.mean(auc_pr_scores, dtype=float)
    vus_roc_avg = np.mean(vus_roc_scores, dtype=float)
    vus_pr_avg = np.mean(vus_pr_scores, dtype=float)
    pate_avg = np.mean(pate_scores, dtype=float)

    auc_roc_std = np.std(auc_roc_scores, dtype=float)
    auc_pr_std = np.std(auc_pr_scores, dtype=float)
    vus_roc_std = np.std(vus_roc_scores, dtype=float)
    vus_pr_std = np.std(vus_pr_scores, dtype=float)
    pate_std = np.std(pate_scores, dtype=float)

    logger.info(f"AUC-ROC: {auc_roc_avg} ± {auc_roc_std}")
    logger.info(f"AUC-PR: {auc_pr_avg} ± {auc_pr_std}")
    logger.info(f"VUS-ROC: {vus_roc_avg} ± {vus_roc_std}")
    logger.info(f"VUS-PR: {vus_pr_avg} ± {vus_pr_std}")
    logger.info(f"PATE: {pate_avg} ± {pate_std}")

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

    evaluation_result = get_metrics(scores, test_label)

    return evaluation_result
