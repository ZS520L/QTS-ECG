"""1-D convolutional variational auto-encoder (reconstruction baseline).

Same convolutional trunk as the AE, with fully connected mu / log-variance
heads on the flattened 256 x 313 feature map (latent dimension 128) and a
KL weight ``beta``. Score: per-record mean squared reconstruction error of a
single stochastic forward pass.

Note: the two flatten -> latent layers and the latent -> flatten layer hold
most of the ~31.5 M parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, in_channels: int = 12, seq_len: int = 5000,
                 channels=(32, 64, 128, 256), latent_dim: int = 128, beta: float = 0.001):
        super().__init__()
        channels = list(channels)
        self.beta = beta
        self.seq_len = seq_len

        enc, ch = [], in_channels
        for c in channels:
            enc += [nn.Conv1d(ch, c, 7, stride=2, padding=3), nn.BatchNorm1d(c), nn.ReLU(True)]
            ch = c
        self.encoder = nn.Sequential(*enc)

        self._enc_len = seq_len
        for _ in channels:
            self._enc_len = (self._enc_len + 1) // 2
        flat_dim = channels[-1] * self._enc_len
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self._dec_ch = channels[-1]

        dec_channels = channels[::-1]
        dec = []
        for i in range(len(dec_channels) - 1):
            dec += [nn.ConvTranspose1d(dec_channels[i], dec_channels[i + 1], 7, stride=2,
                                       padding=3, output_padding=1),
                    nn.BatchNorm1d(dec_channels[i + 1]), nn.ReLU(True)]
        dec.append(nn.ConvTranspose1d(dec_channels[-1], in_channels, 7, stride=2,
                                      padding=3, output_padding=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), self._dec_ch, self._enc_len)
        out = self.decoder(h)
        if out.shape[-1] != self.seq_len:
            out = F.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def compute_loss(self, x):
        recon, mu, logvar = self.forward(x)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        return F.mse_loss(recon, x) + self.beta * kl

    @torch.no_grad()
    def anomaly_score(self, x):
        recon, _, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))
