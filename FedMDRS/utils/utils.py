import os
import numpy as np
from numpy.typing import NDArray
from esn import MDRS
from evaluation.metrics import get_metrics


def train_in_clients(
    serverMachineDataset,
    leaking_rate=1.0,
    rho=0.95,
    delta=0.0001,
    input_scale: float = 1.0,
) -> NDArray:
    N_x = 200

    covariance_matrix = np.zeros((N_x, N_x), dtype=np.float64)
    for serverMachineData in serverMachineDataset:
        local_updates = train_in_client(
            serverMachineData,
            leaking_rate=leaking_rate,
            rho=rho,
            delta=delta,
            input_scale=input_scale,
        )

        covariance_matrix += local_updates

    P_global = np.linalg.inv(covariance_matrix + delta * np.identity(N_x))

    return P_global


def train_in_client(
    serverMachineData,
    leaking_rate=1.0,
    rho=0.95,
    delta=0.0001,
    input_scale: float = 1.0,
) -> NDArray:
    print(f"[train] data name: {serverMachineData.data_name}")
    data_train = serverMachineData.data_train
    N_u = data_train.shape[1]
    N_x = 200
    model = MDRS(
        N_u,
        N_x,
        leaking_rate=leaking_rate,
        delta=delta,
        rho=rho,
        input_scale=input_scale,
    )
    local_updates = model.train(data_train)

    return local_updates


def evaluate_in_clients(
    serverMachineDataset,
    P_global: NDArray,
    output_dir: str,
) -> tuple[float, float, float, float, float]:
    auc_rocs = []
    auc_prs = []
    vus_rocs = []
    vus_prs = []
    pates = []
    for i, serverMachineData in enumerate(serverMachineDataset):
        print(f"Progress Rate: {i / len(serverMachineDataset):.1%}")

        auc_roc, auc_pr, vus_roc, vus_pr, pate = evaluate_in_client(
            serverMachineData, P_global, output_dir
        )
        auc_rocs.append(auc_roc)
        auc_prs.append(auc_pr)
        vus_rocs.append(vus_roc)
        vus_prs.append(vus_pr)
        pates.append(pate)

    auc_roc_avg = np.mean(auc_rocs, dtype=float)
    auc_pr_avg = np.mean(auc_prs, dtype=float)
    vus_roc_avg = np.mean(vus_rocs, dtype=float)
    vus_pr_avg = np.mean(vus_prs, dtype=float)
    pate_avg = np.mean(pates, dtype=float)

    return auc_roc_avg, auc_pr_avg, vus_roc_avg, vus_pr_avg, pate_avg


def evaluate_in_client(
    serverMachineData,
    P: NDArray,
    output_dir: str,
    leaking_rate: float = 1.0,
    rho: float = 0.95,
    delta: float = 0.0001,
    input_scale: float = 1.0,
) -> tuple[float, float, float, float, float]:
    name = serverMachineData.data_name
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    with open(os.path.join(output_dir, name, "log.txt"), "w") as f:
        print(f"[test] dataset name: {name}", file=f)
        print(f"[test] dataset name: {name}")

        label_test = serverMachineData.test_label
        data_test = serverMachineData.data_test

        N_u = data_test.shape[1]
        N_x = 200
        model = MDRS(
            N_u,
            N_x,
            precision_matrix=P,
            leaking_rate=leaking_rate,
            delta=delta,
            rho=rho,
            input_scale=input_scale,
        )
        _, mahalanobis_distances = model.adapt(data_test)

        evaluation_result = get_metrics(mahalanobis_distances, label_test)
        auc_roc = evaluation_result["AUC-ROC"]
        auc_pr = evaluation_result["AUC-PR"]
        vus_roc = evaluation_result["VUS-ROC"]
        vus_pr = evaluation_result["VUS-PR"]
        pate = evaluation_result["PATE"]

    return auc_roc, auc_pr, vus_roc, vus_pr, pate
