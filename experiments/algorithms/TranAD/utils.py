import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from experiments.utils.utils import create_windows


def generate_train_loader(
    train_data,
    batch_size,
    window_size,
    seed=42,
):
    train_data = create_windows(train_data, window_size)
    train_data = shuffle(train_data, random_state=seed)

    # Convert data and labels into PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)

    # Prepare the training data using a TensorDataset (combining data and labels)
    train_data = TensorDataset(
        train_data,
        train_data[:, -1, :],
    )
    # Create DataLoader objects for both training and testing data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    return train_dataloader


def generate_test_loader(
    test_data,
    test_labels,
    batch_size,
    window_size,
):
    test_data = create_windows(test_data, window_size)
    test_labels_point = create_windows(test_labels, window_size)

    # Convert data and labels into PyTorch tensors
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels_point = torch.tensor(test_labels_point, dtype=torch.long)

    # For testing data, shuffling isn't needed, so we just specify the batch size
    test_dataset = TensorDataset(test_data, test_labels_point[:, -1])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader


def getting_labels(data_loader):
    all_labels = []

    for batch in data_loader:
        labels = batch[1].numpy()
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    labels_final = (np.sum(all_labels, axis=1) >= 1) + 0

    return labels_final
