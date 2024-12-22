import os
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerMachineData:
    dataset_name: str
    data_name: str
    data_train: NDArray
    data_test: NDArray
    test_label: NDArray


def create_dataset(
    dataset_name: str = "ServerMachineDataset",
    train_data_dir_path: str = os.path.join(
        "..", "datasets", "ServerMachineDataset", "train"
    ),
    test_data_dir_path: str = os.path.join(
        "..", "datasets", "ServerMachineDataset", "test"
    ),
    test_label_dir_path: str = os.path.join(
        "..", "datasets", "ServerMachineDataset", "test_label"
    ),
) -> list[ServerMachineData]:
    data_filenames = os.listdir(os.path.join(train_data_dir_path))

    dataset: list[ServerMachineData] = []
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
        data = ServerMachineData(
            dataset_name, basename, data_train, data_test, test_label
        )
        dataset.append(data)

    return dataset
