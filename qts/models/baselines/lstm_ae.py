"""LSTM auto-encoder (sequence reconstruction baseline).

The input is average-pooled by 10 (500 steps), encoded by a 2-layer LSTM and
decoded by a 2-layer LSTM initialised with the encoder state, then linearly
interpolated back to 5000 samples. Score: mean squared reconstruction error.

Note: the decoder receives the (down-sampled) input itself as teacher-forcing
input, at both training and test time. This was the evaluated configuration;
it lets the decoder copy its input and explains the below-chance AUROC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAE(nn.Module):
    def __init__(self, in_channels: int = 12, seq_len: int = 5000, hidden_dim: int = 128,
                 num_layers: int = 2, downsample: int = 10):
        super().__init__()
        self.seq_len = seq_len
        self.ds = downsample
        self.encoder = nn.LSTM(in_channels, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(in_channels, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc_out = nn.Linear(hidden_dim, in_channels)

    def forward(self, x):
        x_ds = F.avg_pool1d(x, self.ds).permute(0, 2, 1)          # (B, T, C)
        _, state = self.encoder(x_ds)
        dec_out, _ = self.decoder(x_ds, state)
        recon = self.fc_out(dec_out).permute(0, 2, 1)             # (B, C, T)
        return F.interpolate(recon, size=self.seq_len, mode="linear", align_corners=False)

    def compute_loss(self, x):
        return F.mse_loss(self.forward(x), x)

    @torch.no_grad()
    def anomaly_score(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))
