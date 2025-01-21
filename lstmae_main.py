from logging import getLogger
import os
import numpy as np
from tqdm import trange
from experiments.utils.evaluate import evaluate
from experiments.utils.parser import args_parser
import torch
from torch import nn
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train
from experiments.utils.logger import init_logger
from experiments.utils.utils import get_default_device, set_seed
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train
from experiments.utils.smd import get_SMD_train, get_SMD_test_clients
from experiments.algorithms.LSTMAE.lstmae import LSTMAE
from experiments.algorithms.LSTMAE.utils import (
    generate_test_loader,
    generate_train_loader,
)


def lstmae_main(
    dataset: str,
    result_dir: str,
    loss_fn=nn.MSELoss(),
    optimizer=torch.optim.Adam,
    hidden_size: int = 100,
    batch_size: int = 256,
    epochs: int = 30,
    lr: float = 0.001,
    window_size: int = 100,
    n_layers: tuple = (2, 2),
    use_bias: tuple = (True, True),
    dropout=(0, 0),
    device: str = get_default_device(),
    seed: int = 42,
    evaluate_every: int = 5,
):
    config = locals()
    logger = getLogger(__name__)
    logger.info(config)

    set_seed(seed)

    if dataset == "SMD":
        train_data = get_SMD_train()
        test_clients = get_SMD_test_clients()
    elif dataset == "SMAP":
        train_data = get_SMAP_train()
        test_clients = get_SMAP_test_clients()
    else:
        train_data = get_PSM_train()
        test_clients = get_PSM_test_clients()

    n_features = train_data.shape[1]

    train_dataloader = generate_train_loader(train_data, batch_size, window_size, seed)
    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, batch_size, window_size)
        for test_data, test_labels in test_clients
    ]

    model = LSTMAE(
        loss_fn,
        optimizer,
        n_features=n_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        use_bias=use_bias,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    best_score = 0
    for epoch in trange(epochs):
        model.fit(train_dataloader)

        if (epoch + 1) % evaluate_every == 0:
            score = evaluate(model, test_dataloader_list, result_dir)

            if score > best_score:
                best_score = score

    return best_score


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "lstmae", "centralized", dataset)
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "lstmae.log"))
    logger = getLogger(__name__)

    best_score = np.max([
        lstmae_main(dataset=dataset, result_dir=result_dir, window_size=5),
        lstmae_main(dataset=dataset, result_dir=result_dir, window_size=10),
        lstmae_main(dataset=dataset, result_dir=result_dir, window_size=25),
        lstmae_main(dataset=dataset, result_dir=result_dir, window_size=50),
        lstmae_main(dataset=dataset, result_dir=result_dir, window_size=75),
    ])
    logger.info(f"best score: {best_score}")
