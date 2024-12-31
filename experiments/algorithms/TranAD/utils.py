import os
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from experiments.utils.utils import create_windows


def generate_loaders(
    train_data,
    test_data,
    test_labels,
    batch_size,
    window_size,
    step=1,
    seed=42,
):
    """
    Generate DataLoader objects for training and testing datasets.

    Parameters:
    - train_data: Training dataset.
    - test_data: Testing dataset.
    - test_labels: Labels for the test dataset.
    - seed: Random seed for reproducibility.

    Returns:
    - train_dataloader: DataLoader object for the training dataset.
    - test_dataloader: DataLoader object for the test dataset.
    """

    # Segment the data into overlapping windows
    train_data = create_windows(train_data, window_size, step)
    train_data = shuffle(train_data, random_state=seed)

    test_data = create_windows(test_data, window_size, step)
    test_labels_point = create_windows(test_labels, window_size, step)

    # Convert data and labels into PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels_point = torch.tensor(test_labels_point[:, -1], dtype=torch.long)

    # Print the shapes of the data tensors (useful for debugging and understanding data dimensions)
    print("train window shape: ", train_data.shape)
    print("test window shape: ", test_data.shape)
    print("test window label shape (point-level): ", test_labels_point.shape)

    # Prepare the training data using a TensorDataset (combining data and labels)
    train_data = TensorDataset(
        train_data,
        train_data[:, -1, :],
    )
    # Create DataLoader objects for both training and testing data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # For testing data, shuffling isn't needed, so we just specify the batch size
    test_dataset = TensorDataset(test_data, test_labels_point)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def getting_labels(data_loader):
    all_labels = []

    for batch in data_loader:
        labels = batch[1].numpy()
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    labels_final = (np.sum(all_labels, axis=1) >= 1) + 0

    return labels_final


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


def get_SMD_test(
    test_data_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "test"
    ),
    interpretation_label_dir_path: str = os.path.join(
        os.getcwd(), "datasets", "ServerMachineDataset", "interpretation_label"
    ),
) -> tuple[NDArray, NDArray]:
    data_filenames = os.listdir(test_data_dir_path)

    X_test = []
    y_test = []
    for data_filename in data_filenames:
        test_data_file_path = os.path.join(test_data_dir_path, data_filename)
        interpretation_label_file_path = os.path.join(
            interpretation_label_dir_path, data_filename
        )

        test_data = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
        test_label = get_SMD_interpretation_labels(
            interpretation_label_file_path, test_data.shape
        )

        X_test.append(test_data)
        y_test.append(test_label)

    return np.concatenate(X_test), np.concatenate(y_test)
