from logging import getLogger
import os
from tqdm import trange
from tqdm.contrib import tenumerate
import torch
from torch import nn
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train
from experiments.utils.logger import init_logger
from experiments.utils.utils import get_default_device, set_seed
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train
from experiments.utils.smd import get_SMD_train, get_SMD_test_clients
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.diagram.plot import plot
from experiments.algorithms.USAD.utils import getting_labels
from experiments.algorithms.LSTMAE.lstmae import LSTMAE
from experiments.algorithms.LSTMAE.utils import (
    generate_test_loader,
    generate_train_loader,
)
from experiments.evaluation.metrics import get_metrics


def lstmae_main(
    dataset: str,
    result_dir: str,
    loss_fn=nn.MSELoss(),
    optimizer=torch.optim.Adam,
    hidden_size: int = 100,
    batch_size: int = 256,
    epochs: int = 100,
    lr: float = 0.001,
    window_size: int = 100,
    n_layers: tuple = (2, 2),
    use_bias: tuple = (0, 0),
    dropout=(0, 0),
    device: str = get_default_device(),
    seed: int = 42,
):
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "lstmae.log"))
    logger = getLogger(__name__)
    args = locals()
    logger.info(args)

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
        n_features,
        hidden_size,
        n_layers,
        use_bias,
        dropout,
        batch_size,
        lr,
        device,
    )

    for epoch in trange(epochs):
        model.fit(train_dataloader)

    evaluation_results = []
    for i, test_dataloader in tenumerate(test_dataloader_list):
        scores = model.copy().run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.pdf"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    get_final_scores(evaluation_results, result_dir)


if __name__ == "__main__":
    dataset = "SMAP"
    result_dir = os.path.join("result", "lstmae", "centralized", dataset)

    lstmae_main(dataset=dataset, result_dir=result_dir)
