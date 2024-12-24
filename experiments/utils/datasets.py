from numpy.typing import NDArray
from torch.utils.data import TorchDataset


# For centralized situation
class Dataset(TorchDataset):
    def __init__(self, dataset_name: str, data: NDArray):
        self.data = data
        self.dataset_name = dataset_name

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return self.data.shape[0]


# For federated situation
class Entity(TorchDataset):
    def __init__(
        self,
        dataset_name: str,
        entity_name: str,
        data: NDArray,
    ):
        self.dataset_name = dataset_name
        self.entity_name = entity_name
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return self.data.shape[0]
