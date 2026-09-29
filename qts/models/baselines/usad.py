"""USAD (Audibert et al., KDD 2020), two-decoder adversarial auto-encoder.

Shared convolutional encoder, two decoders. Score:
``0.5 * ||x - AE1(x)||^2 + 0.5 * ||x - AE2(AE1(x))||^2``.

Simplification: a fixed ``alpha = 0.5`` is used instead of the epoch-dependent
schedule, and both adversarial objectives are summed into a single loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class USAD(nn.Module):
    """USAD adapted for 1D ECG signals."""

    def __init__(self, in_channels=12, seq_len=5000,
                 channels=None, latent_dim=128):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        self.seq_len = seq_len
        self._epoch = 0  # track epoch for alpha scheduling

        # Shared encoder
        enc_layers = []
        ch = in_channels
        for c in channels:
            enc_layers += [
                nn.Conv1d(ch, c, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ]
            ch = c
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder 1
        dec_channels = channels[::-1]
        self.decoder1 = self._build_decoder(dec_channels, in_channels, seq_len)
        # Decoder 2
        self.decoder2 = self._build_decoder(dec_channels, in_channels, seq_len)

    def _build_decoder(self, channels, out_channels, seq_len):
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
        return nn.Sequential(*layers)

    def _decode(self, z, decoder):
        out = decoder(z)
        if out.shape[-1] != self.seq_len:
            out = F.interpolate(out, size=self.seq_len, mode="linear", align_corners=False)
        return out

    def forward(self, x):
        """Returns AE1(x), AE2(x), AE2(AE1(x))."""
        z = self.encoder(x)
        ae1 = self._decode(z, self.decoder1)
        ae2 = self._decode(z, self.decoder2)
        # Two-stage: re-encode AE1 output and decode with D2
        z_ae1 = self.encoder(ae1)
        ae2_ae1 = self._decode(z_ae1, self.decoder2)
        return ae1, ae2, ae2_ae1

    def compute_loss(self, x):
        """Combined USAD loss with alpha=0.5 (simplified)."""
        ae1, ae2, ae2_ae1 = self.forward(x)
        alpha = 0.5

        loss1 = alpha * F.mse_loss(ae1, x) + (1 - alpha) * F.mse_loss(ae2_ae1, x)
        loss2 = alpha * F.mse_loss(ae2, x) - (1 - alpha) * F.mse_loss(ae2_ae1.detach(), x)

        return loss1 + loss2

    @torch.no_grad()
    def anomaly_score(self, x):
        ae1, ae2, ae2_ae1 = self.forward(x)
        alpha = 0.5
        score1 = ((x - ae1) ** 2).mean(dim=(1, 2))
        score2 = ((x - ae2_ae1) ** 2).mean(dim=(1, 2))
        return alpha * score1 + (1 - alpha) * score2
