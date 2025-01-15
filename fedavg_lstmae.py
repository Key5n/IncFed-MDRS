import os
import logging
from typing import Dict
from tqdm import trange
from tqdm.contrib import tenumerate
from tqdm.contrib.logging import logging_redirect_tqdm
from experiments.utils.parser import args_parser
import numpy as np
import torch
from torch import nn
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train_clients
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train_clients
from experiments.utils.logger import init_logger
from experiments.algorithms.LSTMAE.lstmae import LSTMAE
from experiments.algorithms.LSTMAE.utils import generate_test_loader
from experiments.algorithms.USAD.utils import getting_labels
from experiments.evaluation.metrics import get_metrics
from experiments.utils.fedavg import calc_averaged_weights
from experiments.utils.utils import choose_clients, get_default_device, set_seed
from experiments.algorithms.LSTMAE.fed_utils import get_clients_LSTMAE
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.diagram.plot import plot
from experiments.utils.smd import get_SMD_test_clients, get_SMD_train_clients


def fedavg_lstmae(
    dataset: str,
    result_dir: str,
    global_epochs: int = 10,
    local_epochs: int = 5,
    client_rate: float = 0.25,
    loss_fn=nn.MSELoss(),
    optimizer_gen_function=torch.optim.Adam,
    hidden_size: int = 128,
    window_size: int = 30,
    n_layers: tuple = (2, 2),
    use_bias: tuple = (True, True),
    dropout: tuple = (0, 0),
    batch_size: int = 256,
    lr: float = 0.001,
    seed: int = 42,
    device=get_default_device(),
):
    args = locals()
    os.makedirs(result_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.info(args)

    set_seed(seed)

    if dataset == "SMD":
        X_train_list = get_SMD_train_clients()
        test_clients = get_SMD_test_clients()
    elif dataset == "SMAP":
        X_train_list = get_SMAP_train_clients()
        test_clients = get_SMAP_test_clients()
    else:
        num_clients = 24
        X_train_list = get_PSM_train_clients(num_clients)
        test_clients = get_PSM_test_clients()

    n_features = X_train_list[0].shape[1]

    clients = get_clients_LSTMAE(
        X_train_list,
        optimizer_gen_function,
        loss_fn,
        local_epochs=local_epochs,
        n_features=n_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        use_bias=use_bias,
        dropout=dropout,
        batch_size=batch_size,
        window_size=window_size,
        lr=lr,
        device=device,
    )

    model = LSTMAE(
        loss_fn,
        optimizer_gen_function,
        n_features,
        hidden_size,
        n_layers,
        use_bias,
        dropout,
        batch_size,
        lr,
        device,
    )
    global_state_dict = model.state_dict()

    with logging_redirect_tqdm():
        for global_round in trange(global_epochs):
            logger.info(global_round)
            # choose different clients for each global round
            # but fix chosen clients on every run for reproduction
            active_clients = choose_clients(clients, client_rate, seed + global_round)

            next_state_dict_list: list[Dict] = []
            data_nums: list[int] = []

            for client in active_clients:
                logger.info(client.client_name)
                next_state_dict, data_num = client.train_avg(global_state_dict)

                next_state_dict_list.append(next_state_dict)
                data_nums.append(data_num)

            global_state_dict = calc_averaged_weights(next_state_dict_list, data_nums)

    model.load_model(global_state_dict)

    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, batch_size, window_size)
        for test_data, test_labels in test_clients
    ]

    evaluation_results = []
    for i, test_dataloader in tenumerate(test_dataloader_list):
        scores = model.copy().run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.pdf"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    get_final_scores(evaluation_results, result_dir)


if __name__ == "__main__":
    args = args_parser()
    dataset = args.dataset
    result_dir = os.path.join("result", "lstmae", "fedavg", dataset)
    init_logger(os.path.join(result_dir, "lstmae.log"))

    fedavg_lstmae(dataset=dataset, result_dir=result_dir)
