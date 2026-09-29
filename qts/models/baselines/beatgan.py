"""BeatGAN (Zhou et al., IJCAI 2019), adversarially regularised reconstruction.

Generator: convolutional encoder/decoder; discriminator: convolutional
classifier. Score: per-record mean squared reconstruction error.

Simplification: generator and discriminator share one optimiser in the
generic trainer; ``compute_loss`` alternates between a generator step
(``lambda_recon * MSE + adversarial``) and a discriminator step on
successive calls (the counter also advances on validation batches).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        layers = []
        ch = in_channels
        for c in channels:
            layers += [
                nn.Conv1d(ch, c, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, out_channels, channels, seq_len):
        super().__init__()
        self.seq_len = seq_len
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.ConvTranspose1d(channels[i], channels[i + 1],
                                   kernel_size=7, stride=2, padding=3, output_padding=1),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers.append(
            nn.ConvTranspose1d(channels[-1], out_channels,
                               kernel_size=7, stride=2, padding=3, output_padding=1)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        out = self.net(z)
        if out.shape[-1] != self.seq_len:
            out = F.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out


class _Discriminator(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        layers = []
        ch = in_channels
        for c in channels:
            layers += [
                nn.Conv1d(ch, c, kernel_size=7, stride=2, padding=3),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = c
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x):
        feat = self.net(x)
        return self.head(feat).squeeze(-1)


class BeatGAN(nn.Module):
    """BeatGAN for ECG anomaly detection.

    Uses adversarial training to improve reconstruction quality.
    Anomaly score = reconstruction MSE (discriminator is auxiliary).
    """

    def __init__(self, in_channels=12, seq_len=5000,
                 channels=None, lambda_recon=10.0):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.lambda_recon = lambda_recon

        self.encoder = _Encoder(in_channels, channels)
        self.decoder = _Decoder(in_channels, channels[::-1], seq_len)
        self.discriminator = _Discriminator(in_channels, channels)

        # Track training phase
        self._train_step = 0

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def compute_loss(self, x):
        """Combined G loss: recon + adversarial.

        For simplicity in the unified trainer, we alternate G/D updates
        internally. Even steps: G update. Odd steps: D update.
        We always return a scalar loss for backward().
        """
        x_recon = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)

        self._train_step += 1
        if self._train_step % 2 == 1:
            # Generator step: fool discriminator + reconstruct well
            d_fake = self.discriminator(x_recon)
            g_adv = F.binary_cross_entropy_with_logits(
                d_fake, torch.ones_like(d_fake))
            return self.lambda_recon * recon_loss + g_adv
        else:
            # Discriminator step
            d_real = self.discriminator(x)
            d_fake = self.discriminator(x_recon.detach())
            d_loss = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                      + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))) / 2
            return d_loss

    @torch.no_grad()
    def anomaly_score(self, x):
        x_recon = self.forward(x)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))
