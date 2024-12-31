from tqdm import tqdm
import torch
from torch import nn
from experiments.algorithms.TranAD.tranad import TranAD
from experiments.evaluation.metrics import get_metrics
from experiments.utils.psm import get_PSM_train, get_PSM_test
from experiments.utils.smd import (
    get_SMD_train,
)
from experiments.algorithms.TranAD.utils import (
    generate_loaders,
    getting_labels,
    set_seed,
    get_SMD_test,
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

    if dataset == "SMD":
        lr = 0.0001

        train_data = get_SMD_train()
        n_features = train_data.shape[1]
        test_data, test_label = get_SMD_test()
    else:
        lr = 0.001

        train_data = get_PSM_train()
        n_features = train_data.shape[1]
        test_data, test_label = get_PSM_test()

    train_dataloader, test_dataloader = generate_loaders(
        train_data,
        test_data,
        test_label,
        batch_size,
        window_size,
        seed=seed,
    )

    model = TranAD(loss_fn, optimizer, scheduler, n_features, lr, batch_size)

    for epoch in tqdm(range(epochs)):
        model.fit(train_dataloader, epoch)
    scores = model.run(test_dataloader)

    labels = getting_labels(test_dataloader)
    evaluation_result = get_metrics(scores, labels)
    print(evaluation_result)
