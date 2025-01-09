import os
from logging import getLogger
from typing import Dict

from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from tqdm import trange
from tqdm.contrib import tenumerate
from experiments.algorithms.USAD.fed_utils import get_SMD_clients_Usad
from experiments.algorithms.USAD.utils import generate_test_loader, getting_labels
from experiments.evaluation.metrics import get_metrics
from experiments.utils.diagram.plot import plot
from experiments.utils.fedavg import calc_averaged_weights
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.smd import get_SMD_test_entities
from experiments.algorithms.USAD.usad import Usad
from experiments.utils.logger import init_logger
from experiments.utils.utils import choose_clients, get_default_device, set_seed

if __name__ == "__main__":
    result_dir = os.path.join("result", "usad", "fedavg")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "usad.log"))
    logger = getLogger(__name__)

    local_epochs = 10
    global_epochs = 100
    client_rate = 0.25

    hidden_size = 100
    device = get_default_device()
    dataset = "SMD"
    seed = 42
    batch_size = 256
    set_seed(seed)

    window_size = 5
    data_channels = 38
    latent_size = 38
    num_epochs = 250

    optimizer = torch.optim.Adam

    w_size = window_size * data_channels
    z_size = window_size * latent_size

    model = Usad(w_size, z_size, optimizer, device)
    global_state_dict = model.state_dict()

    clients = get_SMD_clients_Usad(
        optimizer,
        w_size,
        z_size,
        local_epochs,
        device,
        batch_size,
        window_size,
        seed=seed,
    )

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

    test_entities = get_SMD_test_entities()
    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, window_size, batch_size)
        for test_data, test_labels in test_entities
    ]

    evaluation_results = []
    for i, test_dataloader in tenumerate(test_dataloader_list):
        scores = model.copy().run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.pdf"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    get_final_scores(evaluation_results, result_dir)
