from typing import Dict
import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from experiments.algorithms.LSTMAE.lstmae import LSTMAEModule
from torch.utils.data import DataLoader


# For federated situation
class LSTMAEClient:
    def __init__(
        self,
        entity_name: str,
        train_dataloader: DataLoader,
        optimizer_generate_function,
        loss_fn,
        local_epochs: int,
        n_features: int,
        hidden_size: int,
        n_layers: tuple,
        use_bias: tuple,
        dropout: tuple,
        batch_size: int,
        lr: float,
        device: str,
        prox_mu: float = 0.01,
    ):
        self.entity_name = entity_name
        self.train_dataloader = train_dataloader
        self.optimizer_generate_function = optimizer_generate_function
        self.loss_fn = loss_fn
        self.local_epochs = local_epochs
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.prox_mu = prox_mu

    def train_avg(self, global_state_dict) -> tuple[Dict, int]:
        logger = logging.getLogger(__name__)

        model = LSTMAEModule(
            self.n_features,
            self.hidden_size,
            self.n_layers,
            self.use_bias,
            self.dropout,
            self.device,
        )
        model.load_state_dict(global_state_dict)
        model.to(self.device)
        model.train()

        optimizer = self.optimizer_generate_function(model.parameters(), lr=self.lr)

        with logging_redirect_tqdm():
            for _ in trange(self.local_epochs):
                batch_losses: list[float] = []
                for X, y in self.train_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    _, y_pred = model(X)
                    loss = self.loss_fn(y_pred, y)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_losses.append(loss.item())

                batch_loss_avg = np.sum(batch_losses) / len(batch_losses)
                logger.info(f"loss = {batch_loss_avg}")

        data_num = len(next(iter(self.train_dataloader)))

        return model.state_dict(), data_num

    def train_prox(self, global_state_dict) -> tuple[Dict, int]:
        logger = logging.getLogger(__name__)

        model = LSTMAEModule(
            self.n_features,
            self.hidden_size,
            self.n_layers,
            self.use_bias,
            self.dropout,
            self.device,
        )
        model.load_state_dict(global_state_dict)
        model.to(self.device)
        model.train()
        global_model_parameters = model.parameters

        optimizer = self.optimizer_generate_function(model.parameters(), lr=self.lr)

        with logging_redirect_tqdm():
            for _ in trange(self.local_epochs):
                batch_losses: list[float] = []
                for X, y in self.train_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    _, y_pred = model(X)
                    loss = self.loss_fn(y_pred, y)

                    proximal_term = 0
                    for w, w_0 in zip(model.parameters(), global_model_parameters()):
                        proximal_term += (w - w_0).norm(2)
                    loss += proximal_term * self.prox_mu / 2

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_losses.append(loss.item())

                batch_loss_avg = np.sum(batch_losses) / len(batch_losses)
                logger.info(f"loss = {batch_loss_avg}")

        data_num = len(next(iter(self.train_dataloader)))

        return model.state_dict(), data_num
