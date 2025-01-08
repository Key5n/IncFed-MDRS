import os
import numpy as np
from numpy.typing import NDArray
from experiments.algorithms.TranAD.federated import TranADClient
from experiments.algorithms.TranAD.utils import generate_train_loader


def get_SMD_data(data_dir_path: str) -> list[NDArray]:
    data_filenames = os.listdir(data_dir_path)

    data_list = []
    for data_filename in data_filenames:
        data_file_path = os.path.join(data_dir_path, data_filename)

        data = np.genfromtxt(data_file_path, dtype=np.float64, delimiter=",")

        data_list.append(data)

    return data_list


def get_SMD_clients_TranAD(
    optimizer,
    schedular,
    loss_fn,
    local_epochs: int,
    feats,
    lr,
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
        train_dataloader = generate_train_loader(X_train, batch_size, window_size, seed)
        client = TranADClient(
            f"{dataset_name}-client-{i}",
            train_dataloader,
            optimizer,
            schedular,
            loss_fn,
            local_epochs,
            feats,
            lr,
            device,
        )
        clients.append(client)

    return clients
