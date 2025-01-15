import os
import random
import numpy as np
import torch


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


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def choose_clients(clients: list, client_rate: float, seed) -> list:
    num_active_client = int((len(clients) * client_rate))
    # number of active_clients must be larger than 1
    num_active_client = max(num_active_client, 1)

    rng = np.random.default_rng(seed)
    active_clients_index = rng.choice(
        range(len(clients)), num_active_client, replace=False
    )
    active_clients = [clients[i] for i in active_clients_index]

    return active_clients
