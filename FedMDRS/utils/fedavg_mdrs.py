from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from FedMDRS.utils.utils import train_in_client


def train_in_clients_fedavg(
    train_data_list: list[NDArray],
    N_x: int,
    N_x_tilde: int | None,
    leaking_rate: float,
    rho: float,
    delta: float,
    input_scale: float,
    trans_len: int,
) -> NDArray:
    if N_x_tilde is None:
        N_x_tilde = N_x

    P_global = 1 / delta * np.identity(N_x_tilde)

    all_data_length = np.sum([len(train_data) for train_data in train_data_list])

    for train_data in tqdm(train_data_list):
        local_updates = train_in_client(
            train_data,
            N_x,
            leaking_rate=leaking_rate,
            rho=rho,
            delta=delta,
            input_scale=input_scale,
            N_x_tilde=N_x_tilde,
            trans_len=trans_len,
        )
        local_updates = np.linalg.inv(local_updates)

        P_global += local_updates * len(train_data) / all_data_length

    return P_global
