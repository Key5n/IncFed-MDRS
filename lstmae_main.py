import torch
from torch import nn
from experiments.utils.utils import get_default_device, set_seed
from experiments.utils.psm import get_PSM_train, get_PSM_test
from experiments.utils.smd import (
    get_SMD_test,
    get_SMD_train,
)
from experiments.utils.plot import plot
from experiments.algorithms.LSTMAE.lstmae import LSTMAE
from experiments.algorithms.USAD.utils import getting_labels
from experiments.algorithms.LSTMAE.utils import generate_loaders
from experiments.evaluation.metrics import get_metrics

if __name__ == "__main__":
    hidden_size = 100
    device = get_default_device()
    dataset = "SMD"
    seed = 42
    batch_size = 512
    set_seed(seed)

    if dataset == "SMD":
        epochs = 100
        batch_size = 64
        lr = 0.001
        hidden_size = 128
        window_size = 30
        n_layers = (2, 2)
        use_bias = (True, True)
        dropout = (0, 0)
        step = 100

        train_data = get_SMD_train()
        n_features = train_data.shape[1]
        test_data, test_label = get_SMD_test()
    else:
        epochs = 100
        batch_size = 64
        lr = 0.001
        hidden_size = 128
        window_size = 30
        n_layers = (2, 2)
        use_bias = (True, True)
        dropout = (0, 0)
        step = 100

        train_data = get_PSM_train()
        n_features = train_data.shape[1]
        test_data, test_label = get_PSM_test()

    train_dataloader, test_dataloader = generate_loaders(
        train_data,
        test_data,
        test_label,
        batch_size,
        window_size,
        step,
        seed,
    )

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

    for epoch in range(epochs):
        model.fit(train_dataloader)
        scores = model.run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, f"result/lstmae-{epoch}.png")

        evaluation_result = get_metrics(scores, labels)
        print(evaluation_result)
