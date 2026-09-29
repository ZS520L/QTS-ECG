"""1-D convolutional auto-encoder (reconstruction baseline).

Encoder: 4 x [Conv1d(k=7, s=2) - BN - ReLU]  (12 -> 32 -> 64 -> 128 -> 256 ch)
Decoder: mirrored ConvTranspose1d stack, linear interpolation to 5000 samples.
Score: per-record mean squared reconstruction error.

Note: the bottleneck (256 x 313) is not smaller than the input (12 x 5000),
so the network is not forced to compress; this is the configuration that was
evaluated in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 12, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, ch = [], in_channels
        for c in channels:
            layers += [nn.Conv1d(ch, c, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
            ch = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 12, channels=(256, 128, 64, 32), seq_len: int = 5000):
        super().__init__()
        self.seq_len = seq_len
        layers = []
        for i in range(len(channels) - 1):
            layers += [nn.ConvTranspose1d(channels[i], channels[i + 1], kernel_size=7, stride=2,
                                          padding=3, output_padding=1),
                       nn.BatchNorm1d(channels[i + 1]), nn.ReLU(inplace=True)]
        layers.append(nn.ConvTranspose1d(channels[-1], out_channels, kernel_size=7, stride=2,
                                         padding=3, output_padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        out = self.net(z)
        if out.shape[-1] != self.seq_len:
            out = F.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 12, seq_len: int = 5000, channels=(32, 64, 128, 256)):
        super().__init__()
        channels = list(channels)
        self.encoder = Encoder(in_channels, channels)
        self.decoder = Decoder(in_channels, channels[::-1], seq_len)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def compute_loss(self, x):
        return F.mse_loss(self.forward(x), x)

    @torch.no_grad()
    def anomaly_score(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))
