from numpy.typing import NDArray
from experiments.algorithms.TranAD.federated import TranADClient
from experiments.algorithms.TranAD.utils import generate_train_loader


def get_clients_TranAD(
    X_train_list: list[NDArray],
    optimizer,
    schedular,
    loss_fn,
    local_epochs: int,
    lr: float,
    device: str,
    batch_size: int,
    window_size: int,
    seed=0,
):
    feats = X_train_list[0].shape[1]

    clients = []
    for i, X_train in enumerate(X_train_list):
        train_dataloader = generate_train_loader(X_train, batch_size, window_size, seed)
        client = TranADClient(
            f"client-{i}",
            train_dataloader,
            optimizer,
            schedular,
            loss_fn,
            local_epochs,
            feats,
            lr,
            device,
            window_size
        )
        clients.append(client)

    return clients
