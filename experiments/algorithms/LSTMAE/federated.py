from typing import Dict
from tqdm import tqdm
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

    def train_avg(self, global_state_dict) -> tuple[Dict, int]:
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

        optimizer = self.optimizer_generate_function(model.parameters(), lr=self.lr)

        for _ in tqdm(range(self.local_epochs)):
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
            tqdm.write(f"loss = {batch_loss_avg}")

        data_num = len(next(iter(self.train_dataloader)))

        return model.state_dict(), data_num
