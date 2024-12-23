from .datasets import Entity
import numpy as np
from numpy.typing import NDArray
from mdrs import MDRS
from evaluation.metrics import get_metrics


def train_in_clients(
    entities: list[Entity],
    leaking_rate=1.0,
    rho=0.95,
    delta=0.0001,
    input_scale: float = 1.0,
) -> NDArray:
    N_x = 200

    covariance_matrix = np.zeros((N_x, N_x), dtype=np.float64)
    for entity in entities:
        local_updates = train_in_client(
            entity,
            leaking_rate=leaking_rate,
            rho=rho,
            delta=delta,
            input_scale=input_scale,
        )

        covariance_matrix += local_updates

    P_global = np.linalg.inv(covariance_matrix + delta * np.identity(N_x))

    return P_global


def train_in_client(
    entity: Entity,
    leaking_rate=1.0,
    rho=0.95,
    delta=0.0001,
    input_scale: float = 1.0,
) -> NDArray:
    print(f"[train] data name: {entity.entity_name}")
    data_train = entity.train_data
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


def evaluate(
    X_test: NDArray,
    y_test: NDArray,
    P_global: NDArray,
    leaking_rate: float = 1.0,
    rho: float = 0.95,
    delta: float = 0.0001,
    input_scale: float = 1.0,
) -> tuple[float, float, float, float, float]:

    N_u = X_test.shape[1]
    N_x = 200
    model = MDRS(
        N_u,
        N_x,
        precision_matrix=P_global,
        leaking_rate=leaking_rate,
        delta=delta,
        rho=rho,
        input_scale=input_scale,
    )
    mahalanobis_distances = model.adapt(X_test)

    evaluation_result = get_metrics(mahalanobis_distances, y_test)

    auc_roc = evaluation_result["AUC-ROC"]
    auc_pr = evaluation_result["AUC-PR"]
    vus_roc = evaluation_result["VUS-ROC"]
    vus_pr = evaluation_result["VUS-PR"]
    pate = evaluation_result["PATE"]

    return auc_roc, auc_pr, vus_roc, vus_pr, pate
