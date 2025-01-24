import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler


def get_PSM(data_file_path: str, scale=True) -> NDArray:
    data = pd.read_csv(data_file_path)
    data.drop(columns=[r"timestamp_(min)"], inplace=True)
    data = data.to_numpy()

    if scale:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

    data_without_nan = np.nan_to_num(data)

    return data_without_nan


def get_PSM_train(
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    ),
    proportion: float = 1.0,
) -> NDArray:
    psm_train = get_PSM(train_data_file_path)
    psm_train = psm_train[: int(proportion * len(psm_train))]

    return psm_train


def get_start_index(num_data: int, beta: float, required_length: int, num_clients: int):
    seed = 0
    while True:
        rng = np.random.default_rng(seed)
        # e.g. [1000, 300, 1500, 7200] for num_data is 10000 and num_clients is 4
        proportions = np.floor(rng.dirichlet([beta] * num_clients) * num_data)

        if np.min(proportions) <= required_length:
            seed += 1
            continue

        # [0, 1000, 300, 1500, 7200]
        proportions_start_with_zero = np.concatenate(([0], proportions))
        # [0, 1000, 300, 1500]
        proportions_exclude_last_element = proportions_start_with_zero[:-1]

        # [0, 1000, 1300, 2800], meaning the start index of each client
        start_index = np.floor(np.cumsum(proportions_exclude_last_element)).astype(int)
        break

    return start_index


def get_PSM_list(
    num_clients: int,
    data_file_path: str,
    beta: float | None,
    required_length: int,
    proportion: float,
) -> list[NDArray]:
    data = get_PSM(data_file_path)
    data = data[: int(proportion * len(data))]

    data_list: list[NDArray] = []
    if beta is None:
        data_list = np.array_split(data, num_clients)

        return data_list

    else:
        start_index = get_start_index(len(data), beta, required_length, num_clients)

        for i in range(len(start_index)):
            if i != len(start_index) - 1:
                dataset = data[start_index[i] : start_index[i + 1]]
                data_list.append(dataset)
            else:
                dataset = data[start_index[i] :]
                data_list.append(dataset)

        return data_list


def get_PSM_train_clients(
    num_clients: int,
    proportion: float = 1.0,
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    ),
    beta: float | None = None,
    required_length: int = 10,
) -> list[NDArray]:
    X_train = get_PSM_list(
        num_clients, train_data_file_path, beta, required_length, proportion=proportion
    )

    return X_train


def get_PSM_test_clients(
    test_data_file_path: str = os.path.join(os.getcwd(), "datasets", "PSM", "test.csv"),
    test_label_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "test_label.csv"
    ),
) -> list[tuple[NDArray, NDArray]]:
    test_data = get_PSM(test_data_file_path)
    test_label = get_PSM(test_label_file_path, scale=False)
    test_label = test_label.reshape(-1)

    clients = [(test_data, test_label)]

    return clients
