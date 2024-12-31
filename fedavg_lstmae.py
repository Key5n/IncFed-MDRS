from typing import Dict
import torch
from torch import nn
from experiments.algorithms.LSTMAE.lstmae import LSTMAE, LSTMAEModule
from experiments.algorithms.LSTMAE.utils import generate_test_loader
from experiments.algorithms.USAD.utils import getting_labels
from experiments.evaluation.metrics import get_metrics
from experiments.utils.fedavg import calc_averaged_weights
from experiments.utils.smd import get_SMD_test
from experiments.utils.utils import get_default_device
from experiments.algorithms.LSTMAE.fed_utils import get_SMD_clients_LSTMAE


if __name__ == "__main__":
    device = get_default_device()
    dataset = "SMD"
    seed = 42
    global_epochs = 1
    local_epochs = 1

    loss_fn = nn.MSELoss()
    optimizer_gen_function = torch.optim.Adam
    n_features = 38
    hidden_size = 128
    window_size = 30
    n_layers = (2, 2)
    use_bias = (True, True)
    dropout = (0, 0)
    step = 100
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
    test_data, test_label = get_SMD_test()
    test_dataloader = generate_test_loader(
        test_data, test_label, batch_size, window_size, step
    )

    for global_round in range(global_epochs):
        next_state_dict_list: list[Dict] = []
        data_nums: list[int] = []
        for client in clients:
            next_state_dict, data_num = client.train_avg(global_state_dict)

            next_state_dict_list.append(next_state_dict)
            data_nums.append(data_num)

        global_state_dict = calc_averaged_weights(next_state_dict_list, data_nums)

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
    model.load_model(global_state_dict)

    scores = model.run(test_dataloader)

    labels = getting_labels(test_dataloader)
    evaluation_result = get_metrics(scores, labels)
    print(evaluation_result)
