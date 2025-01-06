import os
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from experiments.utils.utils import create_windows
from experiments.algorithms.LSTMAE.federated import LSTMAEClient


def get_SMD_data(data_dir_path: str) -> list[NDArray]:
    data_filenames = os.listdir(data_dir_path)

    data_list = []
    for data_filename in data_filenames:
        data_file_path = os.path.join(data_dir_path, data_filename)

        data = np.genfromtxt(data_file_path, dtype=np.float64, delimiter=",")

        data_list.append(data)

    return data_list


def get_SMD_clients_LSTMAE(
    optimizer,
    loss_fn,
    local_epochs: int,
    n_features: int,
    hidden_size: int,
    n_layers: tuple,
    use_bias: tuple,
    dropout: tuple,
    batch_size: int,
    lr: float,
    device: str,
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
    window_size=30,
    seed=42,
) -> list[LSTMAEClient]:
    dataset_name = "SMD"
    X_train_list = get_SMD_data(train_data_dir_path)

    clients = []
    for i, X_train in enumerate(X_train_list):
        train_data = create_windows(X_train, window_size)
        train_data = shuffle(train_data, random_state=seed)
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_data = TensorDataset(
            train_data,
            train_data[:, -1, :],
        )
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

        client = LSTMAEClient(
            f"{dataset_name}-entity-{i}",
            train_dataloader,
            optimizer,
            loss_fn,
            local_epochs,
            n_features,
            hidden_size,
            n_layers,
            use_bias,
            dropout,
            batch_size,
            lr,
            device,
        )
        clients.append(client)

    return clients
