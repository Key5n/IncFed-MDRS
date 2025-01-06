import os
from tqdm import tqdm
import torch
from torch import nn
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.plot import plot
from experiments.algorithms.TranAD.tranad import TranAD
from experiments.evaluation.metrics import get_metrics
from experiments.utils.psm import get_PSM_train, get_PSM_test
from experiments.utils.utils import set_seed
from experiments.utils.smd import get_SMD_test_entities, get_SMD_train
from experiments.algorithms.TranAD.utils import (
    generate_test_loader,
    generate_train_loader,
    getting_labels,
)

if __name__ == "__main__":
    dataset = "SMD"
    seed = 42
    batch_size = 128
    epochs = 5
    window_size = 10
    set_seed(seed)

    loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW
    scheduler = torch.optim.lr_scheduler.StepLR

    lr = 0.0001

    train_data = get_SMD_train()
    n_features = train_data.shape[1]
    test_entities = get_SMD_test_entities()

    train_dataloader = generate_train_loader(train_data, batch_size, window_size, seed)
    test_dataloader_list = [
        generate_test_loader(test_data, test_labels, batch_size, window_size)
        for test_data, test_labels in test_entities
    ]

    model = TranAD(loss_fn, optimizer, scheduler, n_features, lr, batch_size)

    for epoch in tqdm(range(epochs)):
        model.fit(train_dataloader, epoch)

    evaluation_results = []
    result_dir = os.path.join("result", "tranad", "centralized")
    os.makedirs(result_dir)
    for i, test_dataloader in enumerate(test_dataloader_list):
        scores = model.run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.png"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    get_final_scores(evaluation_results)
