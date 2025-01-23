from logging import getLogger
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
    trans_len: int,
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
            trans_len=trans_len,
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
    trans_len: int,
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
        trans_len=trans_len,
    )
    local_updates = model.train(train_data)

    return local_updates


def evaluate_in_clients(
    test_data_list: list[tuple[NDArray, NDArray]],
    P_global: NDArray,
    N_x: int,
    leaking_rate: float,
    rho: float,
    delta: float,
    input_scale: float,
    trans_len: int,
    N_x_tilde: int | None = None,
) -> list[Dict]:
    logger = getLogger(__name__)
    evaluation_results: list[Dict] = []

    for test_data, test_label in tqdm(test_data_list):
        evaluation_result = evaluate_in_client(
            test_data,
            test_label,
            P_global=P_global,
            N_x=N_x,
            N_x_tilde=N_x_tilde,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
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


def evaluate_in_client(
    test_data: NDArray,
    test_label: NDArray,
    P_global: NDArray,
    N_x: int,
    leaking_rate: float,
    rho: float,
    delta: float,
    input_scale: float,
    trans_len: int,
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
        trans_len=trans_len,
    )
    scores = model.adapt(test_data)

    scores = scores[trans_len:]
    test_label = test_label[trans_len:]

    evaluation_result = get_metrics(scores, test_label)

    return evaluation_result
