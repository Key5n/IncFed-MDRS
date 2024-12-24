import torch
from .usad import to_device


def evaluate(model, val_loader, n, device):
    outputs = [
        model.validation_step(
            to_device(torch.flatten(batch[0], start_dim=1), device), n
        )
        for batch in val_loader
    ]
    return model.validation_epoch_end(outputs)


def training(
    epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam
):
    history = []
    optimizer1 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder1.parameters())
    )
    optimizer2 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder2.parameters())
    )
    for epoch in range(epochs):
        for batch in train_loader:
            batch = torch.flatten(batch[0], start_dim=1)  # I added this
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch + 1, device)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing_pointwise(model, test_loader, device, alpha=0.5, beta=0.5):
    all_errors = []

    with torch.no_grad():
        for batch in test_loader:
            original_shape = batch[0].shape
            batch_flat = torch.flatten(batch[0], start_dim=1).to(device)

            w1 = model.decoder1(model.encoder(batch_flat))
            w2 = model.decoder2(model.encoder(w1))

            # Reshape reconstructions to original shape
            w1 = w1.view(original_shape)
            w2 = w2.view(original_shape)

            # Move batch[0] to the same device as w1 and w2
            batch_on_device = batch[0].to(device)  # Move batch[0] to the GPU

            # Compute point-wise errors
            errors = (
                alpha * (batch_on_device - w1) ** 2 + beta * (batch_on_device - w2) ** 2
            )

            # Take the mean over the features
            mean_errors = torch.mean(errors, dim=2)

            all_errors.extend(mean_errors.view(-1).tolist())

    return all_errors
