import os
from experiments.algorithms.USAD.federated import UsadClient
from experiments.algorithms.USAD.utils import generate_train_loader
import numpy as np
from numpy.typing import NDArray


def get_SMD_data(data_dir_path: str) -> list[NDArray]:
    data_filenames = os.listdir(data_dir_path)
    data_filenames.sort()

    data_list = []
    for data_filename in data_filenames:
        data_file_path = os.path.join(data_dir_path, data_filename)

        data = np.genfromtxt(data_file_path, dtype=np.float64, delimiter=",")

        data_list.append(data)

    return data_list


def get_SMD_clients_Usad(
    optimizer,
    w_size: int,
    z_size: int,
    local_epochs: int,
    device,
    batch_size,
    window_size,
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
    seed=0,
):
    dataset_name = "SMD"
    X_train_list = get_SMD_data(train_data_dir_path)

    clients = []
    for i, X_train in enumerate(X_train_list):
        train_dataloader = generate_train_loader(X_train, window_size, batch_size, seed)
        client = UsadClient(
            f"{dataset_name}-client-{i}",
            train_dataloader,
            optimizer,
            w_size,
            z_size,
            local_epochs,
            device,
        )
        clients.append(client)

    return clients
