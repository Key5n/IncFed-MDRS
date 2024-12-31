import os
import random
import numpy as np
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
    step,
    seed=42,
):
    """
    Generate DataLoader objects for training and testing datasets.

    Parameters:
    - train_data: Training dataset.
    - test_data: Testing dataset.
    - test_labels: Labels for the test dataset.
    - cfg: Configuration object containing batch size, window size, and other parameters.
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
    test_labels_point = torch.tensor(test_labels_point, dtype=torch.long)

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
    test_dataset = TensorDataset(test_data, test_labels_point[:, -1])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
