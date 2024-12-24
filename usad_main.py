import numpy as np
from experiments.utils.psm import create_PSM_test, create_PSM_train
from experiments.utils.smd import (
    get_SMD_test,
    get_SMD_train,
)
from experiments.algorithms.USAD.usad import get_default_device, to_device, UsadModel
from experiments.algorithms.USAD.utils import (
    generate_loaders,
    set_seed,
    testing_pointwise,
    training,
)
from experiments.evaluation.metrics import get_metrics

if __name__ == "__main__":
    w_size = 5
    hidden_size = 100
    device = get_default_device()
    dataset = "SMD"
    seed = 42
    batch_size = 512
    set_seed(seed)

    if dataset == "SMD":
        step = 5
        window_size = 5
        data_channels = 38
        latent_size = 38
        num_epochs = 250
        anomaly_proportion_window = 0.2
        train_data = get_SMD_train()
        test_data, test_label = get_SMD_test()
    else:
        step = 5
        window_size = 5
        data_channels = 25
        latent_size = 33
        num_epochs = 250
        anomaly_proportion_window = 0.2
        train_data = create_PSM_train()
        test_data = create_PSM_test()

    w_size = window_size * data_channels
    z_size = window_size * latent_size

    train_loader, test_loader = generate_loaders(
        train_data,
        test_data,
        test_label,
        batch_size,
        w_size,
        step,
        anomaly_proportion_window,
    )

    model = UsadModel(w_size * train_data.data.shape[-1], hidden_size)
    model = to_device(model, device)

    history = training(num_epochs, model, train_loader, test_loader, device)
    results_point_wise = testing_pointwise(model, train_loader, device)
    test_rec = np.array(results_point_wise)

    labels = test_data.get_labels()

    evaluation_result = get_metrics(test_rec, labels)
    print(evaluation_result)
