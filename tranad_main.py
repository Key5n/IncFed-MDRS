import os
from tqdm import trange
from tqdm.contrib import tenumerate
import torch
from torch import nn
from experiments.utils.logger import init_logger
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.diagram.plot import plot
from experiments.algorithms.TranAD.tranad import TranAD
from experiments.algorithms.TranAD.smd import get_SMD_test_entities_for_TranAD
from experiments.evaluation.metrics import get_metrics
from experiments.utils.psm import get_PSM_train, get_PSM_test
from experiments.utils.utils import get_default_device, set_seed
from experiments.utils.smd import get_SMD_train
from experiments.algorithms.TranAD.utils import (
    generate_test_loader,
    generate_train_loader,
    getting_labels_for_TranAD,
)

if __name__ == "__main__":
    result_dir = os.path.join("result", "tranad", "centralized")
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "tranad.log"))

    dataset = "SMD"
    seed = 42
    batch_size = 128
    epochs = 5
    window_size = 10
    device = get_default_device()
    set_seed(seed)

    loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW
    scheduler = torch.optim.lr_scheduler.StepLR

    lr = 0.0001

    train_data = get_SMD_train()
    n_features = train_data.shape[1]
    test_entities = get_SMD_test_entities_for_TranAD()

    train_dataloader = generate_train_loader(train_data, batch_size, window_size, seed)
    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, batch_size, window_size)
        for test_data, test_labels in test_entities
    ]

    model = TranAD(loss_fn, optimizer, scheduler, n_features, lr, batch_size, device)

    for epoch in trange(epochs):
        model.fit(train_dataloader, epoch)

    evaluation_results = []
    for i, test_dataloader in tenumerate(test_dataloader_list):
        scores = model.run(test_dataloader)
        labels = getting_labels_for_TranAD(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.pdf"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    get_final_scores(evaluation_results, result_dir)
