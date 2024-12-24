import os
import pandas as pd
import numpy as np
from .datasets import Entity, Dataset


def create_PSM_entities(
    num_entities: int,
    data_file_path: str,
    beta: float | None,
    required_length: int,
) -> list[Entity]:
    dataset_name = "PSM"
    data = pd.read_csv(data_file_path)
    data.drop(columns=[r"timestamp_(min)"], inplace=True)
    data = data.to_numpy()

    entities: list[Entity] = []
    if beta is None:
        datasets = np.array_split(data, num_entities)

        for i, dataset in enumerate(datasets):
            entity = Entity(dataset_name, f"entity-{i}", dataset)
            entities.append(entity)

        return entities

    else:
        while True:
            num_data = data.shape[0]

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
            start_index = np.floor(
                np.cumsum(proportions_exclude_last_element) * num_data
            )
            break

        for i in range(len(start_index)):
            if i != len(start_index) - 1:
                dataset = data[start_index[i] : start_index[i + 1]]
                entity = Entity(dataset_name, f"entity-{i}", dataset)
                entities.append(entity)
            else:
                dataset = data[start_index[i] :]
                entity = Entity(dataset_name, f"entity-{i}", dataset)
                entities.append(entity)

        return entities


def create_PSM(data_file_path: str) -> Dataset:
    dataset_name = "PSM"
    data = pd.read_csv(data_file_path)
    psm = Dataset(dataset_name, data)

    return psm


def create_PSM_train(
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    )
) -> Dataset:
    psm_train = create_PSM(train_data_file_path)

    return psm_train


def create_PSM_test(
    test_data_file_path: str = os.path.join(os.getcwd(), "datasets", "PSM", "test.csv")
) -> Dataset:
    psm_test = create_PSM(test_data_file_path)

    return psm_test


def create_PSM_entities_train(
    num_entites: int,
    train_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "train.csv"
    ),
    beta: float | None = None,
    required_length: int = 100,
) -> list[Entity]:
    train_entities = create_PSM_entities(
        num_entites, train_data_file_path, beta, required_length
    )

    return train_entities


def create_PSM_entities_test(
    num_entites: int,
    test_data_file_path: str = os.path.join(
        os.getcwd(), "datasets", "PSM", "test_label.csv"
    ),
    beta: float | None = None,
    required_length: int = 100,
) -> list[Entity]:
    test_entities = create_PSM_entities(
        num_entites, test_data_file_path, beta, required_length
    )

    return test_entities
