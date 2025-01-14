import os
import numpy as np
from numpy.typing import NDArray


def get_SMD_interpretation_labels(
    interpretation_label_file_path: str, shape: tuple
) -> NDArray:
    label = np.zeros(shape)
    with open(os.path.join(interpretation_label_file_path), "r") as f:
        lines = f.readlines()

    for line in lines:
        pos, values = line.split(":")[0], line.split(":")[1].split(",")
        start, end, index = (
            int(pos.split("-")[0]),
            int(pos.split("-")[1]),
            [int(i) - 1 for i in values],
        )
        label[start - 1 : end - 1, index] = 1

    return label


def get_SMD_test_entities_for_TranAD(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    interpretation_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "interpretation_label"
    ),
) -> list[tuple[NDArray, NDArray]]:
    data_filenames = os.listdir(test_data_dir_path)

    entities: list[tuple[NDArray, NDArray]] = []
    for data_filename in data_filenames:
        test_data_file_path = os.path.join(test_data_dir_path, data_filename)
        interpretation_label_file_path = os.path.join(
            interpretation_label_dir_path, data_filename
        )

        test_data = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
        test_label = get_SMD_interpretation_labels(
            interpretation_label_file_path, test_data.shape
        )
        test_label = (np.sum(test_label, axis=1) >= 1) + 0

        if test_data.shape[0] != test_label.shape[0]:
            raise Exception(f"Length mismatch while creating SMD test dataset")

        entity = (test_data, test_label)
        entities.append(entity)

    return entities
