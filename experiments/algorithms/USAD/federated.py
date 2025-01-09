from logging import getLogger
import numpy as np
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from experiments.algorithms.USAD.usad import UsadModule
from experiments.utils.utils import to_device
import torch
from torch.utils.data import DataLoader


class UsadClient:
    def __init__(
        self,
        client_name: str,
        train_dataloader: DataLoader,
        optimizer_generate_function,
        w_size: int,
        z_size: int,
        local_epochs: int,
        device: str,
    ):
        self.client_name = client_name
        self.train_dataloader = train_dataloader
        self.optimizer_generate_function = optimizer_generate_function
        self.w_size = w_size
        self.z_size = z_size
        self.local_epochs = local_epochs
        self.device = device

    def train_avg(self, global_state_dict):
        logger = getLogger(__name__)

        model = UsadModule(self.w_size, self.z_size)
        model.load_state_dict(global_state_dict)
        model.to(self.device)

        optimizer1 = self.optimizer_generate_function(
            list(model.encoder.parameters()) + list(model.decoder1.parameters())
        )
        optimizer2 = self.optimizer_generate_function(
            list(model.encoder.parameters()) + list(model.decoder2.parameters())
        )

        model.train()

        for epoch in trange(self.local_epochs):
            loss1_list = []
            loss2_list = []
            for batch in self.train_dataloader:
                batch = torch.flatten(batch[0], start_dim=1)  # I added this
                batch = to_device(batch, self.device)

                # Train AE1
                loss1, loss2 = model.training_step(batch, epoch + 1)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())

                # Train AE2
                loss1, loss2 = model.training_step(batch, epoch + 1)

                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()

                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())

            with logging_redirect_tqdm():
                logger.info(
                    f"Epoch {epoch}, loss1: {np.mean(loss1_list)}, loss2: {np.mean(loss2_list)}"
                )

        data_num = len(next(iter(self.train_dataloader)))
        return model.state_dict(), data_num
