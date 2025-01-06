import os
import logging
from typing import Dict
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.plot import plot
import numpy as np
import torch
from torch import nn
from experiments.utils.logger import init_logger
from experiments.algorithms.LSTMAE.lstmae import LSTMAE, LSTMAEModule
from experiments.algorithms.LSTMAE.utils import generate_test_loader
from experiments.algorithms.USAD.utils import getting_labels
from experiments.evaluation.metrics import get_metrics
from experiments.utils.fedavg import calc_averaged_weights
from experiments.utils.smd import get_SMD_test_entities
from experiments.utils.utils import choose_clients, get_default_device, set_seed
from experiments.algorithms.LSTMAE.fed_utils import get_SMD_clients_LSTMAE


if __name__ == "__main__":
    result_dir = os.path.join("result", "lstmae", "federated")
    init_logger(os.path.join(result_dir, f"{__file__}.log"))
    logger = logging.getLogger(__name__)

    device = get_default_device()
    dataset = "SMD"
    seed = 42
    global_epochs = 10
    local_epochs = 5
    client_rate = 0.25
    set_seed()

    loss_fn = nn.MSELoss()
    optimizer_gen_function = torch.optim.Adam
    n_features = 38
    hidden_size = 128
    window_size = 30
    n_layers = (2, 2)
    use_bias = (True, True)
    dropout = (0, 0)
    batch_size = 512
    lr = 0.001

    model = LSTMAEModule(n_features, hidden_size, n_layers, use_bias, dropout, device)
    global_state_dict = model.state_dict()

    clients = get_SMD_clients_LSTMAE(
        optimizer_gen_function,
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

    test_entities = get_SMD_test_entities()
    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, batch_size, window_size)
        for test_data, test_labels in test_entities
    ]
    # for testing
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

    for global_round in range(global_epochs):
        logger.info(global_round)
        # choose different clients for each global round
        # but fix chosen clients on every run for reproduction
        active_clients = choose_clients(clients, client_rate, seed + global_round)

        next_state_dict_list: list[Dict] = []
        data_nums: list[int] = []

        for client in active_clients:
            logger.info(client.entity_name)
            next_state_dict, data_num = client.train_avg(global_state_dict)

            next_state_dict_list.append(next_state_dict)
            data_nums.append(data_num)

        global_state_dict = calc_averaged_weights(next_state_dict_list, data_nums)

        model.load_model(global_state_dict)

        evaluation_results = []
        for i, test_dataloader in enumerate(test_dataloader_list):
            scores = model.run(test_dataloader)
            labels = getting_labels(test_dataloader)

            plot(scores, labels, os.path.join(result_dir, f"{i}.png"))

            evaluation_result = get_metrics(scores, labels)
            evaluation_results.append(evaluation_result)

        get_final_scores(evaluation_results)
