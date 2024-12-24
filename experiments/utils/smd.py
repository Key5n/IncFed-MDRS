import os
import numpy as np
from numpy.typing import NDArray
from .datasets import Dataset, Entity


def get_SMD(data_dir_path: str) -> NDArray:
    data_filenames = os.listdir(data_dir_path)

    dataset = []
    for data_filename in data_filenames:
        data_file_path = os.path.join(data_dir_path, data_filename)

        entity = np.genfromtxt(data_file_path, dtype=np.float64, delimiter=",")
        dataset.append(entity)
    concatenated_dataset = np.concatenate(dataset)

    return concatenated_dataset


def get_SMD_train(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
) -> NDArray:
    X_train = get_SMD(train_data_dir_path)

    return X_train


def get_SMD_test(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    test_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test_label"
    ),
) -> tuple[NDArray, NDArray]:
    X_test = get_SMD(test_data_dir_path)
    y_test = get_SMD(test_label_dir_path)
    if X_test.shape[0] != y_test.shape[0]:
        raise Exception(f"Length mismatch while creating SMD test dataset")

    return X_test, y_test


def get_SMD_data(data_dir_path: str) -> NDArray:
    data_filenames = os.listdir(data_dir_path)

    data_list = []
    for data_filename in data_filenames:
        data_file_path = os.path.join(data_dir_path, data_filename)

        data = np.genfromtxt(data_file_path, dtype=np.float64, delimiter=",")

        data_list.append(data)

    return np.array(data_list)


def get_SMD_entities_train(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
) -> list[Entity]:
    dataset_name = "SMD"
    X_train_list = get_SMD_data(train_data_dir_path)

    entities = []
    for i, X_train in enumerate(X_train_list):
        entity = Entity(dataset_name, f"entity-{i}", X_train, None)
        entities.append(entity)

    return entities


def get_SMD_entities_test(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    test_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test_label"
    ),
) -> list[Entity]:
    dataset_name = "SMD"
    X_test_list = get_SMD_data(test_data_dir_path)
    y_test_list = get_SMD_data(test_label_dir_path)

    entities = []
    for i, (X_test, y_test) in enumerate(zip(X_test_list, y_test_list)):
        entity = Entity(dataset_name, f"entity-{i}", X_test, y_test)
        entities.append(entity)

    return entities
