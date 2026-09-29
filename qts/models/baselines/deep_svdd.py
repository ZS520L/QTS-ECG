"""Deep SVDD (Ruff et al., ICML 2018), one-class representation baseline.

Encoder: 3 x [Conv1d(k=7, s=2) - BN - LeakyReLU(0.2)] -> global average
pooling -> Linear(128). The hypersphere centre ``c`` is the mean embedding of
the training data at initialisation (coordinates with ``|c_j| < 0.1`` are
pushed to +-0.1) and is kept fixed. Score: ``||phi(x) - c||^2``.

Note: unlike the original method, the encoder keeps bias terms and batch
normalisation (a known risk of hypersphere collapse); early stopping on the
validation loss limits the effect in practice.
"""

import torch
import torch.nn as nn


class DeepSVDD(nn.Module):
    def __init__(self, in_channels: int = 12, seq_len: int = 5000,
                 channels=(32, 64, 128), rep_dim: int = 128):
        super().__init__()
        channels = list(channels)
        self.rep_dim = rep_dim
        enc, ch = [], in_channels
        for c in channels:
            enc += [nn.Conv1d(ch, c, 7, stride=2, padding=3), nn.BatchNorm1d(c),
                    nn.LeakyReLU(0.2, True)]
            ch = c
        enc.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*enc)
        self.fc = nn.Linear(channels[-1], rep_dim)
        self.register_buffer("center", torch.zeros(rep_dim))

    def phi(self, x):
        return self.fc(self.encoder(x).squeeze(-1))

    @torch.no_grad()
    def initialize(self, loader, device, eps: float = 0.1) -> None:
        """Set ``c`` to the mean training embedding (network in eval mode)."""
        self.eval()
        c = torch.cat([self.phi(x.to(device)) for x, _ in loader]).mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.center.copy_(c)

    def compute_loss(self, x):
        return ((self.phi(x) - self.center) ** 2).sum(dim=1).mean()

    @torch.no_grad()
    def anomaly_score(self, x):
        return ((self.phi(x) - self.center) ** 2).sum(dim=1)
