from numpy.typing import NDArray
from torch.utils.data import TorchDataset


# For centralized situation
class Dataset(TorchDataset):
    def __init__(self, dataset_name: str, data: NDArray, label: NDArray | None):
        self.data = data
        # if label is None, set data to label
        self.label = label
        self.dataset_name = dataset_name

    def __getitem__(self, i):
        if self.label is None:
            return self.data[i], None
        else:
            return self.data[i], self.label[i]

    def __len__(self):
        return self.data.shape[0]

    def get_labels(self) -> NDArray | None:
        return self.label


# For federated situation
class Entity(TorchDataset):
    def __init__(
        self,
        dataset_name: str,
        entity_name: str,
        data: NDArray,
        label: NDArray | None,
    ):
        self.dataset_name = dataset_name
        self.entity_name = entity_name
        self.data = data
        self.label = label

    def __getitem__(self, i):
        if self.label is None:
            return self.data[i], None
        else:
            return self.data[i], self.label[i]

    def __len__(self):
        return self.data.shape[0]
