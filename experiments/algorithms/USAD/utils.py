import os
import random
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from .usad import to_device
from tqdm import tqdm


def evaluate(model, val_loader, n, device):
    outputs = [
        model.validation_step(
            to_device(torch.flatten(batch[0], start_dim=1), device), n
        )
        for batch in val_loader
    ]
    return model.validation_epoch_end(outputs)


def training(
    epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam
):
    history = []
    optimizer1 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder1.parameters())
    )
    optimizer2 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder2.parameters())
    )
    for epoch in tqdm(range(epochs)):
        for batch in train_loader:
            batch = torch.flatten(batch[0], start_dim=1)  # I added this
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch + 1, device)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing_pointwise(model, test_loader, device, alpha=0.5, beta=0.5):
    all_errors = []

    with torch.no_grad():
        for batch in test_loader:
            original_shape = batch[0].shape
            batch_flat = torch.flatten(batch[0], start_dim=1).to(device)

            w1 = model.decoder1(model.encoder(batch_flat))
            w2 = model.decoder2(model.encoder(w1))

            # Reshape reconstructions to original shape
            w1 = w1.view(original_shape)
            w2 = w2.view(original_shape)

            # Move batch[0] to the same device as w1 and w2
            batch_on_device = batch[0].to(device)  # Move batch[0] to the GPU

            # Compute point-wise errors
            errors = (
                alpha * (batch_on_device - w1) ** 2 + beta * (batch_on_device - w2) ** 2
            )

            # Take the mean over the features
            mean_errors = torch.mean(errors, dim=2)

            all_errors.extend(mean_errors.view(-1).tolist())

    return all_errors


# Function to create windows from the data
def create_windows(data, window_size, step=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i : i + window_size])
    return np.array(windows)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def generate_loaders(
    train_data,
    test_data,
    test_labels,
    batch_size,
    window_size,
    step,
    anomaly_proportion_window,
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

    # Create dummy labels for training data to match its shape
    dummy_train_labels_point = np.zeros_like(train_data, dtype=int)
    dummy_train_labels_window = np.zeros((train_data.shape[0],), dtype=int)

    test_data = create_windows(test_data, window_size, step)
    test_labels_point = create_windows(test_labels, window_size, step)

    # Label windows as anomalous if the proportion of anomalous points within them exceeds a threshold
    test_labels_window = (
        np.mean(test_labels_point, axis=1) > anomaly_proportion_window
    ).astype(int)

    # Convert data and labels into PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels_point = torch.tensor(test_labels_point, dtype=torch.long)
    test_labels_window = torch.tensor(test_labels_window, dtype=torch.long)

    # Print the shapes of the data tensors (useful for debugging and understanding data dimensions)
    print("train window shape: ", train_data.shape)
    print("test window shape: ", test_data.shape)
    print("test window label shape (point-level): ", test_labels_point.shape)
    print("test window label shape (window-level): ", test_labels_window.shape)

    # Prepare the training data using a TensorDataset (combining data and labels)
    train_data = TensorDataset(
        train_data,
        torch.tensor(dummy_train_labels_point, dtype=torch.long),
        torch.tensor(dummy_train_labels_window, dtype=torch.long),
    )
    # Create DataLoader objects for both training and testing data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # For testing data, shuffling isn't needed, so we just specify the batch size
    test_dataset = TensorDataset(test_data, test_labels_point, test_labels_window)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
