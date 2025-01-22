from experiments.utils.scaffold import (
    get_client_update,
    update_client_control_variate,
    update_model_with_control_variates,
)
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
        window_size: int,
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
        self.window_size = window_size
        self.device = device

        # fedprox
        self.prox_mu = prox_mu

        # scaffold
        initial_model = TranADModule(feats, lr, window_size=window_size)
        initial_model.to(device)
        self.c_local = initial_model.state_dict()

    def train_avg(self, global_state_dict) -> tuple[Dict, int]:
        logger = getLogger(__name__)

        model = TranADModule(self.feats, self.lr, window_size=self.window_size)
        model.load_state_dict(global_state_dict)
        model.to(self.device)
        model.train()

        optimizer = self.optimizer_generate_function(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        schedular = self.schedular(optimizer, 5, 0.9)

        for epoch in trange(self.local_epochs):
            l1s = []
            n = epoch + 1
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
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            schedular.step()
            with logging_redirect_tqdm():
                logger.info(f"Epoch {n},\tL1 = {np.mean(l1s)}")

        data_num = len(next(iter(self.train_dataloader)))
        return model.state_dict(), data_num

    def train_prox(self, global_state_dict) -> tuple[Dict, int]:
        logger = getLogger(__name__)

        model = TranADModule(self.feats, self.lr, window_size=self.window_size)
        model.load_state_dict(global_state_dict)
        global_model_parameters = model.parameters()

        model.to(self.device)
        model.train()

        optimizer = self.optimizer_generate_function(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        schedular = self.schedular(optimizer, 5, 0.9)

        for epoch in trange(self.local_epochs):
            l1s = []
            n = epoch + 1
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

    def train_scaffold(
        self, global_state_dict: Dict, c_global: Dict
    ) -> tuple[Dict, int, Dict]:
        logger = getLogger()

        model = TranADModule(self.feats, self.lr, window_size=self.window_size)
        model.load_state_dict(global_state_dict)
        model.to(self.device)
        model.train()

        optimizer = self.optimizer_generate_function(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        schedular = self.schedular(optimizer, 5, 0.9)

        count = 0
        with logging_redirect_tqdm():
            for epoch in trange(self.local_epochs):

                n = epoch + 1
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

                    current_model_dict = model.state_dict()
                    updated_model = update_model_with_control_variates(
                        current_model_dict, c_global, self.c_local, self.lr
                    )

                    model.load_state_dict(updated_model)
                    count += 1

                schedular.step()
                with logging_redirect_tqdm():
                    logger.info(f"Epoch {n},\tL1 = {np.mean(l1s)}")

        next_c_local = update_client_control_variate(
            model.state_dict(),
            global_state_dict,
            self.c_local,
            c_global,
            count,
            self.lr,
        )

        c_local_updates = get_client_update(next_c_local, self.c_local)
        self.c_local = next_c_local

        data_num = len(next(iter(self.train_dataloader)))
        return model.state_dict(), data_num, c_local_updates
