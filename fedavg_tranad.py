from logging import getLogger
import os
from typing import Dict
from tqdm import trange
from tqdm.contrib import tenumerate
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from torch import nn
from experiments.algorithms.TranAD.fed_utils import get_clients_TranAD
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train_clients
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train_clients
from experiments.utils.logger import init_logger
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.diagram.plot import plot
from experiments.algorithms.TranAD.tranad import TranAD
from experiments.algorithms.TranAD.smd import get_SMD_test_entities_for_TranAD
from experiments.evaluation.metrics import get_metrics
from experiments.utils.utils import choose_clients, get_default_device, set_seed
from experiments.algorithms.TranAD.utils import generate_test_loader
from experiments.utils.fedavg import calc_averaged_weights
from experiments.algorithms.USAD.utils import getting_labels
from experiments.utils.smd import get_SMD_train_clients


def fedavg_tranad(
    dataset: str,
    result_dir: str,
    global_epochs=10,
    local_epochs=5,
    client_rate=0.25,
    seed=42,
    batch_size=128,
    window_size=10,
    lr=0.0001,
    loss_fn=nn.MSELoss(reduction="none"),
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.StepLR,
    device=get_default_device(),
):
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "tranad.log"))
    logger = getLogger(__name__)
    args = locals()
    logger.info(args)

    set_seed(seed)

    if dataset == "SMD":
        X_train_list = get_SMD_train_clients()
        test_clients = get_SMD_test_entities_for_TranAD()
    elif dataset == "SMAP":
        X_train_list = get_SMAP_train_clients()
        test_clients = get_SMAP_test_clients()
    else:
        num_clients = 24
        X_train_list = get_PSM_train_clients(num_clients)
        test_clients = get_PSM_test_clients()

    n_features = X_train_list[0].shape[1]

    clients = get_clients_TranAD(
        X_train_list,
        optimizer,
        scheduler,
        loss_fn,
        local_epochs=local_epochs,
        lr=lr,
        device=device,
        batch_size=batch_size,
        window_size=window_size,
        seed=seed,
    )

    model = TranAD(
        loss_fn, optimizer, scheduler, n_features, lr, batch_size, window_size, device
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
    dataset = "SMD"
    result_dir = os.path.join("result", "tranad", "fedavg", dataset)

    fedavg_tranad(dataset=dataset, result_dir=result_dir)
