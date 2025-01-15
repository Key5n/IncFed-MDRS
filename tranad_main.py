from logging import getLogger
import os
from tqdm import trange
from tqdm.contrib import tenumerate
import torch
from torch import nn
from experiments.algorithms.USAD.utils import getting_labels
from experiments.utils.psm import get_PSM_test_clients, get_PSM_train
from experiments.utils.smap import get_SMAP_test_clients, get_SMAP_train
from experiments.utils.logger import init_logger
from experiments.utils.get_final_scores import get_final_scores
from experiments.utils.diagram.plot import plot
from experiments.algorithms.TranAD.tranad import TranAD
from experiments.algorithms.TranAD.smd import get_SMD_test_entities_for_TranAD
from experiments.evaluation.metrics import get_metrics
from experiments.utils.utils import get_default_device, set_seed
from experiments.utils.smd import get_SMD_train
from experiments.algorithms.TranAD.utils import (
    generate_test_loader,
    generate_train_loader,
)


def tranad_main(
    dataset: str,
    result_dir: str,
    seed: int = 42,
    batch_size: int = 128,
    epochs: int = 5,
    window_size: int = 10,
    device: str = get_default_device(),
    loss_fn=nn.MSELoss(reduction="none"),
    optimizer=torch.optim.AdamW,
    scheduler: int = torch.optim.lr_scheduler.StepLR,
    lr: float = 0.0001,
):
    os.makedirs(result_dir, exist_ok=True)
    init_logger(os.path.join(result_dir, "tranad.log"))
    logger = getLogger(__name__)
    args = locals()
    logger.info(args)

    set_seed(seed)

    if dataset == "SMD":
        train_data = get_SMD_train()
        test_clients = get_SMD_test_entities_for_TranAD()
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

    model = TranAD(
        loss_fn, optimizer, scheduler, n_features, lr, batch_size, window_size, device
    )

    for epoch in trange(epochs):
        model.fit(train_dataloader, epoch)

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
    result_dir = os.path.join("result", "tranad", "centralized", dataset)

    tranad_main(dataset=dataset, result_dir=result_dir)
