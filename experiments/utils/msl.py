import os
import ast
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler


def get_MSL_list(data_dir_path: str, label_file_path) -> list[NDArray]:
    labeled_anomalies = pd.read_csv(label_file_path)
    labeled_anomalies.sort_values("chan_id", inplace=True)
    smap_entity_ids = labeled_anomalies[labeled_anomalies["spacecraft"] == "MSL"][
        "chan_id"
    ].values

    data_list = []
    for smap_entity_id in smap_entity_ids:
        data_file_path = os.path.join(data_dir_path, f"{smap_entity_id}.npy")

        with open(data_file_path, "rb") as f:
            data = np.load(f)

            scaler = MinMaxScaler()
            data_transformed = scaler.fit_transform(data)

            data_list.append(data_transformed[:, 0].reshape((-1, 1)))
            # data_list.append(data_transformed)

    return data_list


def get_MSL_concatenated(data_dir_path: str, label_file_path: str) -> NDArray:
    dataset = get_MSL_list(data_dir_path, label_file_path)
    dataset_concatenated = np.concatenate(dataset)

    # return dataset_concatenated[:, 0].reshape((-1, 1))
    return dataset_concatenated[:, 0].reshape((-1, 1))


def get_MSL_train(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "SMAP_MSL", "train"
    ),
    label_file_path: str = os.path.join(
        os.getcwd(), "datasets", "SMAP_MSL", "labeled_anomalies.csv"
    ),
) -> NDArray:
    train_data = get_MSL_concatenated(train_data_dir_path, label_file_path)

    return train_data


def get_MSL_train_clients(
    train_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "SMAP_MSL", "train"
    ),
    label_file_path: str = os.path.join(
        os.getcwd(), "datasets", "SMAP_MSL", "labeled_anomalies.csv"
    ),
) -> list[NDArray]:
    train_data_list = get_MSL_list(train_data_dir_path, label_file_path)

    return train_data_list


def get_MSL_test_label(label_file_path: str):
    labeled_anomalies = pd.read_csv(label_file_path)
    labeled_anomalies.sort_values("chan_id", inplace=True)

    smap_entities = labeled_anomalies[labeled_anomalies["spacecraft"] == "MSL"]

    test_labels = []
    for i, smap_entity in smap_entities.iterrows():
        test_label = np.zeros(smap_entity["num_values"])

        anomaly_sequences = smap_entity["anomaly_sequences"]
        anomaly_sequences = ast.literal_eval(anomaly_sequences)
        for sequence in anomaly_sequences:
            test_label[sequence[0] : (sequence[1] + 1)] = 1

        test_labels.append(test_label)

    return test_labels


def get_MSL_test_clients(
    test_data_dir_path: str = os.path.join(os.getcwd(), "datasets", "SMAP_MSL", "test"),
    label_file_path: str = os.path.join(
        os.getcwd(), "datasets", "SMAP_MSL", "labeled_anomalies.csv"
    ),
) -> list[tuple[NDArray, NDArray]]:
    clients: list[tuple[NDArray, NDArray]] = []

    test_data_list = get_MSL_list(test_data_dir_path, label_file_path)
    test_label_list = get_MSL_test_label(label_file_path)

    for test_data, test_label in zip(test_data_list, test_label_list):
        if test_data.shape[0] != test_label.shape[0]:
            raise Exception(f"Length mismatch while creating SMD test dataset")
        clients.append((test_data, test_label))

    return clients
