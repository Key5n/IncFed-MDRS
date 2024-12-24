import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from .datasets import Entity, Dataset


def get_PSM(data_file_path: str) -> Dataset:
    data = pd.read_csv(data_file_path)
    data.drop(columns=[r"timestamp_(min)"], inplace=True)
    data = data.to_numpy()

    return data


def get_PSM_train(
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    )
) -> Dataset:
    psm_train = get_PSM(train_data_file_path)

    return psm_train


def get_PSM_test(
    test_data_file_path: str = os.path.join(os.getcwd(), "datasets", "PSM", "test.csv")
) -> Dataset:
    psm_test = get_PSM(test_data_file_path)

    return psm_test


def get_start_index(
    num_data: int, beta: float, required_length: int, num_entities: int
):
    while True:
        # e.g. [1000, 300, 1500, 7200] for num_data is 10000 and num_entities is 4
        proportions = np.floor(
            np.random.dirichlet(np.repeat(beta, num_entities)) * num_data
        )

        if np.min(proportions) <= required_length:
            continue

        # [0, 1000, 300, 1500, 7200]
        proportions_start_with_zero = np.concatenate(([0], proportions))
        # [0, 1000, 300, 1500]
        proportions_exclude_last_element = proportions_start_with_zero[:-1]

        # [0, 1000, 1300, 2800], meaning the start index of each client
        start_index = np.floor(np.cumsum(proportions_exclude_last_element) * num_data)
        break

    return start_index


def get_PSM_data(
    num_entities: int,
    data_file_path: str,
    beta: float | None,
    required_length: int,
) -> NDArray:
    data = pd.read_csv(data_file_path)
    data.drop(columns=[r"timestamp_(min)"], inplace=True)
    data = data.to_numpy()

    data_list: list[NDArray] = []
    if beta is None:
        data_list = np.array_split(data, num_entities)

        return data_list

    else:
        start_index = get_start_index(data, beta, required_length, num_entities)

        for i in range(len(start_index)):
            if i != len(start_index) - 1:
                dataset = data[start_index[i] : start_index[i + 1]]
                data_list.append(dataset)
            else:
                dataset = data[start_index[i] :]
                data_list.append(dataset)

        return data_list


def get_PSM_entities_train(
    num_entites: int,
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    ),
    beta: float | None = None,
    required_length: int = 100,
) -> NDArray:
    X_train = get_PSM_data(num_entites, train_data_file_path, beta, required_length)

    return X_train


def get_PSM_entities_test(
    num_entities: int,
    test_data_file_path: str = os.path.join(os.getcwd(), "datasets", "PSM", "test.csv"),
    test_label_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "test_label.csv"
    ),
    beta: float | None = None,
    required_length: int = 100,
) -> tuple[NDArray, NDArray]:
    test_data = pd.read_csv(test_data_file_path)
    test_data.drop(columns=[r"timestamp_(min)"], inplace=True)
    test_data = test_data.to_numpy()

    test_label = pd.read_csv(test_label_file_path)
    test_label.drop(columns=[r"timestamp_(min)"], inplace=True)
    test_label = test_data.to_numpy()

    if beta is None:
        X_test = np.array_split(test_data, num_entities)
        y_test = np.array_split(test_label, num_entities)
    else:
        start_index = get_start_index(test_data, beta, required_length, num_entities)

        X_test: list[NDArray] = []
        y_test: list[NDArray] = []
        for i in range(len(start_index)):
            if i != len(start_index) - 1:
                test_dataset = test_data[start_index[i] : start_index[i + 1]]
                test_label = test_label[start_index[i] : start_index[i + 1]]
                X_test.append(test_dataset)
                y_test.append(test_label)
            else:
                test_dataset = test_data[start_index[i] :]
                test_label = test_label[start_index[i] :]
                X_test.append(test_dataset)
                y_test.append(test_label)

    return X_test, y_test
