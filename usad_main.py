import os
import torch
from tqdm import trange
from tqdm.contrib import tenumerate
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.logger import init_logger
from experiments.utils.diagram.plot import plot
from experiments.utils.smd import get_SMD_test_entities, get_SMD_train
from experiments.algorithms.USAD.usad import Usad
from experiments.algorithms.USAD.utils import (
    generate_test_loader,
    generate_train_loader,
    getting_labels,
)
from experiments.utils.utils import get_default_device, set_seed
from experiments.evaluation.metrics import get_metrics


if __name__ == "__main__":
    result_dir = os.path.join("result", "usad", "centralized")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "usad.log"))

    hidden_size = 100
    device = get_default_device()
    dataset = "SMD"
    seed = 42
    batch_size = 512
    set_seed(seed)

    window_size = 5
    data_channels = 38
    latent_size = 38
    num_epochs = 250
    train_data = get_SMD_train()

    optimizer = torch.optim.Adam

    train_dataloader = generate_train_loader(
        train_data, window_size, batch_size, seed=seed
    )
    test_entities = get_SMD_test_entities()
    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, window_size, batch_size)
        for test_data, test_labels in test_entities
    ]

    w_size = window_size * data_channels
    z_size = window_size * latent_size

    model = Usad(w_size, z_size, optimizer, device)

    for epoch in trange(num_epochs):
        model.fit(train_dataloader, epoch)

    evaluation_results = []
    for i, test_dataloader in tenumerate(test_dataloader_list):
        scores = model.run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.png"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    get_final_scores(evaluation_results, result_dir)
