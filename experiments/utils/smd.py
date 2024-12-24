import os
import numpy as np
from .datasets import Dataset, Entity


def create_SMD(data_dir_path: str) -> Dataset:
    dataset_name = "SMD"
    data_filenames = os.listdir(data_dir_path)

    dataset = []
    for data_filename in data_filenames:
        data_file_path = os.path.join(data_dir_path, data_filename)

        entity = np.genfromtxt(data_file_path, dtype=np.float64, delimiter=",")
        dataset.append(entity)
    concatenated_dataset = np.concatenate(dataset)

    smd = Dataset(dataset_name, concatenated_dataset)
    return smd


def create_SMD_train(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    )
):
    smd_train = create_SMD(train_data_dir_path)

    return smd_train


def create_SMD_test(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    )
):
    smd_test = create_SMD(test_data_dir_path)

    return smd_test


def create_SMD_entities(data_dir_path: str) -> list[Entity]:
    dataset_name = "SMD"
    data_filenames = os.listdir(data_dir_path)

    entities: list[Entity] = []
    for data_filename in data_filenames:
        train_data_file_path = os.path.join(data_dir_path, data_filename)

        train_data = np.genfromtxt(
            train_data_file_path, dtype=np.float64, delimiter=","
        )

        entity_name = data_filename.split(".")[0]

        entity = Entity(dataset_name, entity_name, train_data)
        entities.append(entity)

    return entities


def create_SMD_entities_train(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
) -> list[Entity]:
    train_entities = create_SMD_entities(train_data_dir_path)

    return train_entities


def create_SMD_entities_test(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "train"
    ),
) -> list[Entity]:
    test_entities = create_SMD_entities(test_data_dir_path)

    return test_entities
