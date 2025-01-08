import os
from tqdm import trange
import torch
from torch import nn
from experiments.utils.logger import init_logger
from experiments.utils.utils import get_default_device, set_seed
from experiments.utils.psm import get_PSM_train, get_PSM_test
from experiments.utils.smd import (
    get_SMD_test_entities,
    get_SMD_train,
)
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.plot import plot
from experiments.algorithms.LSTMAE.lstmae import LSTMAE
from experiments.algorithms.USAD.utils import getting_labels
from experiments.algorithms.LSTMAE.utils import (
    generate_test_loader,
    generate_train_loader,
)
from experiments.evaluation.metrics import get_metrics

if __name__ == "__main__":
    result_dir = os.path.join("result", "lstmae", "centralized")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, f"{__file__}.log"))

    hidden_size = 100
    device = get_default_device()
    dataset = "SMD"
    seed = 42
    batch_size = 512
    set_seed(seed)

    epochs = 100
    batch_size = 64
    lr = 0.001
    hidden_size = 128
    window_size = 30
    n_layers = (2, 2)
    use_bias = (True, True)
    dropout = (0, 0)

    train_data = get_SMD_train()
    n_features = train_data.shape[1]
    test_entities = get_SMD_test_entities()

    train_dataloader = generate_train_loader(train_data, batch_size, window_size, seed)
    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, batch_size, window_size)
        for test_data, test_labels in test_entities
    ]

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam

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
    for i, test_dataloader in enumerate(test_dataloader_list):
        scores = model.run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.png"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    filename = os.path.join(result_dir, "boxplot.png")
    get_final_scores(evaluation_results, filename)
