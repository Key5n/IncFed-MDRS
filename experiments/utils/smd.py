import os
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler


def get_SMD_list(data_dir_path: str, scale=True) -> list[NDArray]:
    data_filenames = os.listdir(data_dir_path)

    data_list = []
    for data_filename in data_filenames:
        data_file_path = os.path.join(data_dir_path, data_filename)
        data = np.genfromtxt(data_file_path, dtype=np.float64, delimiter=",")

        if scale:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)

        data_list.append(data)

    return data_list


def get_SMD_concatenated(data_dir_path: str) -> NDArray:
    dataset = get_SMD_list(data_dir_path)
    concatenated_dataset = np.concatenate(dataset)

    return concatenated_dataset


def get_SMD_train(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
) -> NDArray:
    X_train = get_SMD_concatenated(train_data_dir_path)

    return X_train


def get_SMD_train_clients(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    )
) -> list[NDArray]:
    X_train_list = get_SMD_list(train_data_dir_path)

    return X_train_list


def get_SMD_test_clients(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    test_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test_label"
    ),
) -> list[tuple[NDArray, NDArray]]:
    clients: list[tuple[NDArray, NDArray]] = []

    test_data_list = get_SMD_list(test_data_dir_path)
    test_label_list = get_SMD_list(test_label_dir_path, scale=False)

    for test_data, test_label in zip(test_data_list, test_label_list):
        if test_data.shape[0] != test_label.shape[0]:
            raise Exception(f"Length mismatch while creating SMD test dataset")
        clients.append((test_data, test_label))

    return clients
