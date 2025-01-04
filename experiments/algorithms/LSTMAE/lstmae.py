import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm


class LSTMAE:
    def __init__(
        self,
        loss_fn,
        optimizer,
        n_features: int,
        hidden_size: int,
        n_layers: tuple,
        use_bias: tuple,
        dropout: tuple,
        batch_size: int,
        lr: float,
        device: str,
    ):
        self.model = LSTMAEModule(
            n_features, hidden_size, n_layers, use_bias, dropout, device
        )
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size

        self.model.to_device(device)

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def fit(self, dataloader):
        self.model.train()

        losses = []
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            _, y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())

        loss_avg = np.mean(losses)
        tqdm.write(f"loss: {loss_avg}")

    def run(self, dataloader):
        self.model.eval()

        scores = []
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                score, _ = self.model(X)
                scores.append(score.detach().cpu().numpy())
        scores_np = np.concatenate(scores)

        return scores_np


class LSTMAEModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: tuple,
        use_bias: tuple,
        dropout: tuple,
        device: str,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.device = device

        self.encoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[0],
            bias=self.use_bias[0],
            dropout=self.dropout[0],
        )
        self.decoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[1],
            bias=self.use_bias[1],
            dropout=self.dropout[1],
        )
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)

    def _init_hidden(self, batch_size):
        return (
            self.to_var(
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()
            ),
            self.to_var(
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()
            ),
        )

    def to_var(self, t, **kwargs):
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, device):
        self.to(device)

    def forward(self, ts_batch):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(
            ts_batch.float(), enc_hidden
        )  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(
                    ts_batch[:, i].unsqueeze(1).float(), dec_hidden
                )
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        output = output[:, -1]
        rec_error = nn.L1Loss(reduction="none")(output, ts_batch[:, -1])

        rec_error_mean = torch.mean(rec_error, dim=1)

        return rec_error_mean, output
