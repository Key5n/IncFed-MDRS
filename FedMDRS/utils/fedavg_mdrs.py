from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from FedMDRS.utils.utils import train_in_client


def train_in_clients_fedavg(
    train_data_list: list[NDArray],
    N_x: int = 200,
    leaking_rate=1.0,
    rho=0.95,
    delta=0.0001,
    input_scale: float = 1.0,
) -> NDArray:

    covariance_matrix = np.zeros((N_x, N_x), dtype=np.float64)

    all_data_length = np.sum([len(train_data) for train_data in train_data_list])

    for train_data in tqdm(train_data_list):
        local_updates = train_in_client(
            train_data,
            N_x,
            leaking_rate=leaking_rate,
            rho=rho,
            delta=delta,
            input_scale=input_scale,
        )

        covariance_matrix += local_updates * len(train_data) / all_data_length

    P_global = np.linalg.inv(covariance_matrix + delta * np.identity(N_x))

    return P_global
