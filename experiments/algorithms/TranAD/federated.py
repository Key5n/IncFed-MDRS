import numpy as np
from typing import Dict
from logging import getLogger
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.data import DataLoader
import torch

from experiments.algorithms.TranAD.tranad import TranADModule


class TranADClient:
    def __init__(
        self,
        client_name: str,
        train_dataloader: DataLoader,
        optimizer_generate_function,
        schedular,
        loss_fn,
        local_epochs: int,
        feats,
        lr,
        device,
        prox_mu: float = 0.01,
    ):
        self.client_name = client_name
        self.train_dataloader = train_dataloader
        self.optimizer_generate_function = optimizer_generate_function
        self.schedular = schedular
        self.loss_fn = loss_fn
        self.local_epochs = local_epochs
        self.feats = feats
        self.lr = lr
        self.device = device

        self.prox_mu = prox_mu

    def train_avg(self, global_state_dict) -> tuple[Dict, int]:
        logger = getLogger()

        model = TranADModule(self.feats, self.lr)
        model.load_state_dict(global_state_dict)
        model.to(self.device)
        model.train()

        optimizer = self.optimizer_generate_function(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        schedular = self.schedular(optimizer, 5, 0.9)

        for n in trange(self.local_epochs):
            l1s = []
            for d, _ in self.train_dataloader:
                d = d.to(self.device)

                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, self.feats)
                z = model(window, elem)
                l1 = (
                    self.loss_fn(z, elem)
                    if not isinstance(z, tuple)
                    else (1 / n) * self.loss_fn(z[0], elem)
                    + (1 - 1 / n) * self.loss_fn(z[1], elem)
                )
                if isinstance(z, tuple):
                    z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

            schedular.step()
            with logging_redirect_tqdm():
                logger.info(f"Epoch {n},\tL1 = {np.mean(l1s)}")

        data_num = len(next(iter(self.train_dataloader)))
        return model.state_dict(), data_num

    def train_prox(self, global_state_dict) -> tuple[Dict, int]:
        logger = getLogger()

        model = TranADModule(self.feats, self.lr)
        model.load_state_dict(global_state_dict)
        global_model_parameters = model.parameters()

        model.to(self.device)
        model.train()

        optimizer = self.optimizer_generate_function(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        schedular = self.schedular(optimizer, 5, 0.9)

        for n in trange(self.local_epochs):
            l1s = []
            for d, _ in self.train_dataloader:
                d = d.to(self.device)

                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, self.feats)
                z = model(window, elem)
                l1 = (
                    self.loss_fn(z, elem)
                    if not isinstance(z, tuple)
                    else (1 / n) * self.loss_fn(z[0], elem)
                    + (1 - 1 / n) * self.loss_fn(z[1], elem)
                )
                if isinstance(z, tuple):
                    z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)

                proximal_term = 0
                for w, w_0 in zip(model.parameters(), global_model_parameters()):
                    proximal_term += (w - w_0).norm(2)
                loss += proximal_term * self.prox_mu / 2

                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

            schedular.step()
            with logging_redirect_tqdm():
                logger.info(f"Epoch {n},\tL1 = {np.mean(l1s)}")

        data_num = len(next(iter(self.train_dataloader)))
        return model.state_dict(), data_num
