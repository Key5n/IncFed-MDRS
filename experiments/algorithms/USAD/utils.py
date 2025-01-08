from tqdm import tqdm
import numpy as np
import torch
from experiments.utils.utils import create_windows, to_device
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle


def generate_train_loader(train_data, window_size, batch_size, seed=42):
    # Segment the data into overlapping windows
    train_data = create_windows(train_data, window_size)
    train_data = shuffle(train_data, random_state=seed)

    # Prepare the training data using a TensorDataset (combining data and labels)
    train_data = TensorDataset(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    return train_dataloader


def generate_test_loader(test_data, test_labels, window_size, batch_size, seed=42):
    test_data = create_windows(test_data, window_size)
    test_labels_point = create_windows(test_labels, window_size)

    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels_point = torch.tensor(test_labels_point, dtype=torch.long)

    # For testing data, shuffling isn't needed, so we just specify the batch size
    test_dataset = TensorDataset(test_data, test_labels_point)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader


def getting_labels(data_loader):
    all_labels = []

    for batch in data_loader:
        labels = batch[1].numpy()
        all_labels.extend(labels.flatten())

    # Convert list to numpy array
    all_labels = np.array(all_labels)

    return all_labels
