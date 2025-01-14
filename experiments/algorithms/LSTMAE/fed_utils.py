from numpy.typing import NDArray
from experiments.algorithms.LSTMAE.utils import generate_train_loader
from experiments.algorithms.LSTMAE.federated import LSTMAEClient

def get_clients_LSTMAE(
    X_train_list: list[NDArray],
    optimizer,
    loss_fn,
    local_epochs: int,
    n_features: int,
    hidden_size: int,
    n_layers: tuple,
    use_bias: tuple,
    dropout: tuple,
    batch_size: int,
    window_size: int,
    lr: float,
    device: str,
    seed: int = 42
) -> list[LSTMAEClient]:
    clients = []
    for i, X_train in enumerate(X_train_list):
        train_dataloader = generate_train_loader(
            X_train,
            batch_size=batch_size,
            window_size=window_size,
            seed=seed
        )

        client = LSTMAEClient(
            f"client-{i}",
            train_dataloader,
            optimizer,
            loss_fn,
            local_epochs,
            n_features,
            hidden_size,
            n_layers,
            use_bias,
            dropout,
            batch_size,
            lr,
            device,
        )
        clients.append(client)

    return clients
