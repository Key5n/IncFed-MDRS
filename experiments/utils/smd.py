import os
import numpy as np
from numpy.typing import NDArray


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


def get_SMD_test_entities(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    test_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test_label"
    ),
) -> list[tuple[NDArray, NDArray]]:
    data_filenames = os.listdir(test_data_dir_path)

    entities: list[tuple[NDArray, NDArray]] = []
    for data_filename in data_filenames:
        test_data_file_path = os.path.join(test_data_dir_path, data_filename)
        test_label_file_path = os.path.join(test_label_dir_path, data_filename)

        test_data = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
        test_label = np.genfromtxt(
            test_label_file_path, dtype=np.float64, delimiter=","
        )
        if test_data.shape[0] != test_label.shape[0]:
            raise Exception(f"Length mismatch while creating SMD test dataset")

        entity: tuple[NDArray, NDArray] = (test_data, test_label)
        entities.append(entity)

    return entities
