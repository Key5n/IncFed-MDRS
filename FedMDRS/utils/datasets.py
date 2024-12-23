import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    dataset_name: str
    entity_name: str
    train_data: NDArray


def get_dataset(dataset_name: str) -> tuple[list[Entity], NDArray, NDArray]:
    if dataset_name == "SMD":
        entities = create_SMD_train()
        X_test, y_test = create_SMD_test()
    elif dataset_name == "PSM":
        entities = create_PSM_train()
        X_test, y_test = create_SMD_test()
    else:
        raise Exception(f"Dataset {dataset_name} is unknown")

    return entities, X_test, y_test


def create_SMD_train(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
) -> list[Entity]:
    dataset_name = "SMD"
    data_filenames = os.listdir(os.path.join(train_data_dir_path))

    entities: list[Entity] = []
    for data_filename in data_filenames:
        train_data_file_path = os.path.join(train_data_dir_path, data_filename)

        data_train = np.genfromtxt(
            train_data_file_path, dtype=np.float64, delimiter=","
        )

        basename = data_filename.split(".")[0]
        data = Entity(dataset_name, basename, data_train)
        entities.append(data)

    return entities


def create_SMD_test(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    test_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test_label"
    ),
) -> tuple[NDArray, NDArray]:
    data_filenames = os.listdir(os.path.join(test_data_dir_path))

    X_test = []
    y_test = []
    for data_filename in data_filenames:
        test_data_file_path = os.path.join(test_data_dir_path, data_filename)
        test_label_file_path = os.path.join(test_label_dir_path, data_filename)

        data_test = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
        test_label = np.genfromtxt(
            test_label_file_path, dtype=np.float64, delimiter=","
        )
        X_test.append(data_test)
        y_test.append(test_label)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return X_test, y_test


def create_PSM_train(
    num_clients: int = 10,
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    ),
) -> list[Entity]:
    train_data = pd.read_csv(train_data_file_path)
    train_data.drop(columns=[r"timestamp_(min)"], inplace=True)
    train_data = train_data.to_numpy()

    entities = []
    each_train_length = train_data // num_clients
    for i in range(num_clients):
        train_data_entity = train_data[
            i * each_train_length : (i + 1) * each_train_length
        ]

        entity = Entity("PSM", f"PSM-{i}", train_data_entity)
        entities.append(entity)

    return entities


def create_PSM_test(
    test_data_file_path: str = os.path.join(os.getcwd(), "datasets", "PSM", "test.csv"),
    test_label_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "test_label.csv"
    ),
) -> tuple[NDArray, NDArray]:

    X_test = pd.read_csv(test_data_file_path).drop(
        columns=[r"timestamp_(min)"], inplace=True
    )
    X_test = X_test.values.to_numpy()
    y_test = pd.read_csv(test_label_file_path).drop(
        columns=[r"timestamp_(min)"], inplace=True
    )
    y_test = y_test.values.to_numpy()

    return X_test, y_test
