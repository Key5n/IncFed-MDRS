import numpy as np
from torch.utils.data import DataLoader
from .usad import UsadModel, get_default_device, to_device
from experiments.utils.psm import create_PSM_test, create_PSM_train
from experiments.utils.smd import create_SMD_test, create_SMD_train
from experiments.algorithms.USAD.utils import testing_pointwise, training
from experiments.evaluation.metrics import get_metrics

if __name__ == "__main__":
    w_size = 5
    hidden_size = 100
    device = get_default_device()
    dataset = "SMD"

    if dataset == "SMD":
        train_data = create_SMD_train()
        test_data = create_SMD_test()
        window_size = 5
        data_channels = 38
        latent_size = 38
        num_epochs = 250
    else:
        train_data = create_PSM_train()
        test_data = create_PSM_test()
        window_size = 5
        data_channels = 25
        latent_size = 33
        num_epochs = 250

    w_size = window_size * data_channels
    z_size = window_size * latent_size

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = UsadModel(w_size * train_data.data.shape[-1], hidden_size)
    model = to_device(model, device)

    history = training(num_epochs, model, train_loader, test_loader, device)
    results_point_wise = testing_pointwise(model, train_loader, device)
    test_rec = np.array(results_point_wise)

    labels = test_data.get_labels()

    evaluation_result = get_metrics(test_rec, labels)
