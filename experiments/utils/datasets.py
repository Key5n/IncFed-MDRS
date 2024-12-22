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
    test_data: NDArray
    test_label: NDArray


def create_SMD(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    test_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test_label"
    ),
) -> list[Entity]:
    dataset_name = "SMD"
    data_filenames = os.listdir(os.path.join(train_data_dir_path))

    entities: list[Entity] = []
    for data_filename in data_filenames:
        train_data_file_path = os.path.join(train_data_dir_path, data_filename)
        test_data_file_path = os.path.join(test_data_dir_path, data_filename)
        test_label_file_path = os.path.join(test_label_dir_path, data_filename)

        data_train = np.genfromtxt(
            train_data_file_path, dtype=np.float64, delimiter=","
        )
        data_test = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
        test_label = np.genfromtxt(
            test_label_file_path, dtype=np.float64, delimiter=","
        )

        basename = data_filename.split(".")[0]
        data = Entity(dataset_name, basename, data_train, data_test, test_label)
        entities.append(data)

    return entities


def create_PSM(
    num_clients: int = 10,
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    ),
    test_data_file_path: str = os.path.join(os.getcwd(), "datasets", "PSM", "test.csv"),
    test_label_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "test_label.csv"
    ),
) -> list[Entity]:
    train_data = pd.read_csv(train_data_file_path)
    train_data.drop(columns=[r"timestamp_(min)"], inplace=True)

    test_data = pd.read_csv(test_data_file_path)
    test_label = pd.read_csv(test_label_file_path)

    train_data = train_data.to_numpy()

    entities = []
    each_train_length = train_data // num_clients
    each_test_length = test_data // num_clients
    for i in range(num_clients):
        train_data_entity = train_data[
            i * each_train_length : (i + 1) * each_train_length
        ]
        test_data_entity = test_data[i * each_test_length : (i + 1) * each_test_length]
        test_label_entity = test_label[
            i * each_test_length : (i + 1) * each_test_length
        ]

        entity = Entity(
            "PSM", f"PSM-{i}", train_data_entity, test_data_entity, test_label_entity
        )
        entities.append(entity)

    return entities
