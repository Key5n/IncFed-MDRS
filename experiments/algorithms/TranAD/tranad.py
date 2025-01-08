from logging import getLogger
import math

from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranADModule(nn.Module):
    def __init__(self, feats, lr):
        super(TranADModule, self).__init__()
        self.name = "TranAD"
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranAD:
    def __init__(self, loss_fn, optimizer, scheduler, feats, lr, batch_size, device):
        self.feats = feats
        self.model = TranADModule(feats, lr)
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = scheduler(self.optimizer, 5, 0.9)
        self.batch_size = batch_size
        self.device = device
        self.model.to(device)

    def fit(self, dataloader, epoch):
        logger = getLogger(__name__)
        n = epoch + 1
        l1s = []
        for d, _ in dataloader:
            d = d.to(self.device)

            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, self.feats)
            z = self.model(window, elem)
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
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        self.scheduler.step()
        with logging_redirect_tqdm():
            logger.info(f"Epoch {epoch},\tL1 = {np.mean(l1s)}")
        return np.mean(l1s), self.optimizer.param_groups[0]["lr"]

    def run(self, dataloader):
        self.model.eval()

        losses = []
        with torch.no_grad():
            for d, _ in dataloader:
                d = d.to(self.device)

                local_batch_size = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_batch_size, self.feats)
                z = self.model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]
                loss = self.loss_fn(z, elem)[0]
                losses.append(loss.detach().cpu().numpy())

        losses_concatenated = np.concatenate(losses)
        loss_final = np.mean(losses_concatenated, axis=1)

        return loss_final

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
