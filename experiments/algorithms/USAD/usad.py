import numpy as np
from logging import getLogger
from tqdm import trange
from experiments.utils.utils import to_device
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean(
            (batch - w3) ** 2
        )
        return loss1, loss2

    def validation_step(self, batch, n):
        with torch.no_grad():
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)
            w3 = self.decoder2(self.encoder(w1))
            loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean(
                (batch - w3) ** 2
            )
            loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean(
                (batch - w3) ** 2
            )
        return {"val_loss1": loss1, "val_loss2": loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x["val_loss1"] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x["val_loss2"] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {"val_loss1": epoch_loss1.item(), "val_loss2": epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(
                epoch, result["val_loss1"], result["val_loss2"]
            )
        )


class Usad(nn.Module):
    def __init__(self, w_size, z_size, optimizer, device: str):
        self.model = UsadModel(w_size, z_size)
        self.optimizer1 = optimizer(
            self.model.encoder.parameters() + list(self.model.decoder1.parameters())
        )
        self.optimizer2 = optimizer(
            self.model.encoder.parameters() + list(self.model.decoder2.parameters())
        )
        self.device = device

        self.model = to_device(self.model, device)

    def fit(self, dataloader, epoch) -> None:
        logger = getLogger(__name__)

        loss1_list = []
        loss2_list = []
        for batch in dataloader:
            batch = torch.flatten(batch[0], start_dim=1)  # I added this
            batch = to_device(batch, self.device)

            # Train AE1
            loss1, loss2 = self.model.training_step(batch, epoch + 1)
            loss1.backward()
            self.optimizer1.step()
            self.optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = self.model.training_step(batch, epoch + 1)

            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())

            loss2.backward()
            self.optimizer2.step()
            self.optimizer2.zero_grad()

        logger.info(
            f"Epoch {epoch}, loss1: {np.mean(loss1_list)}, loss2: {np.mean(loss2_list)}"
        )

    def run(self, dataloader, alpha=0.5, beta=0.5):
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for batch in dataloader:
                original_shape = batch[0].shape
                batch_flat = torch.flatten(batch[0], start_dim=1).to(self.device)

                w1 = self.model.decoder1(self.model.encoder(batch_flat))
                w2 = self.model.decoder2(self.model.encoder(w1))

                # Reshape reconstructions to original shape
                w1 = w1.view(original_shape)
                w2 = w2.view(original_shape)

                # Move batch[0] to the same device as w1 and w2
                batch_on_device = batch[0].to(self.device)  # Move batch[0] to the GPU

                # Compute point-wise errors
                errors = (
                    alpha * (batch_on_device - w1) ** 2
                    + beta * (batch_on_device - w2) ** 2
                )

                # Take the mean over the features
                mean_errors = torch.mean(errors, dim=2)

                all_errors.extend(mean_errors.view(-1).tolist())

        return np.array(all_errors)
