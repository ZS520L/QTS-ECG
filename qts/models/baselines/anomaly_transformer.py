"""Anomaly Transformer (Xu et al., ICLR 2022), association-discrepancy baseline.

The recording is cut into 100 patches of 50 samples x 12 leads (600-d tokens)
projected to ``d_model``. Each layer computes a *series* association
(softmax self-attention) and a *prior* association (Gaussian kernel over
token distance with a learnable width per head). Score: reconstruction error
+ ``lambda_ad`` x symmetric KL between the two associations.

Simplification: the minimax (stop-gradient) two-phase optimisation of the
original is replaced by minimising ``recon - lambda_ad * discrepancy``, and
the score adds the discrepancy instead of using ``softmax(-AD) * recon``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.sigma = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, x):
        B, T, _ = x.shape
        H, dk = self.n_heads, self.d_k
        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)

        series = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk), dim=-1)

        pos = torch.arange(T, device=x.device).float()
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        sigma = self.sigma.abs().clamp(min=0.1)
        prior = (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (dist / sigma) ** 2)
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        prior = prior.unsqueeze(0).expand(B, -1, -1, -1)

        out = torch.matmul(series, V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out), series, prior


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads, seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        a, series, prior = self.attn(x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x, series, prior


class AnomalyTransformer(nn.Module):
    def __init__(self, in_channels: int = 12, seq_len: int = 5000, patch_size: int = 50,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3, d_ff: int = 256,
                 lambda_ad: float = 1.0):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.lambda_ad = lambda_ad
        self.seq_len = seq_len
        self.in_channels = in_channels
        patch_dim = in_channels * patch_size
        self.input_proj = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.layers = nn.ModuleList([
            AnomalyTransformerBlock(d_model, n_heads, self.n_patches, d_ff)
            for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, patch_dim)

    def _patchify(self, x):
        B, C, L = x.shape
        P = self.patch_size
        N = L // P
        x = x[:, :, :N * P].reshape(B, C, N, P)
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, C * P)

    def _unpatchify(self, patches):
        B, N, _ = patches.shape
        C, P = self.in_channels, self.patch_size
        return patches.view(B, N, C, P).permute(0, 2, 1, 3).contiguous().view(B, C, N * P)

    def forward(self, x):
        h = self.input_proj(self._patchify(x)) + self.pos_embed
        series_list, prior_list = [], []
        for layer in self.layers:
            h, s, p = layer(h)
            series_list.append(s)
            prior_list.append(p)
        recon = self._unpatchify(self.output_proj(h))
        if recon.shape[-1] != self.seq_len:
            recon = F.interpolate(recon, size=self.seq_len, mode="linear", align_corners=False)
        return recon, series_list, prior_list

    @staticmethod
    def _association_discrepancy(series_list, prior_list):
        ad = 0.0
        for series, prior in zip(series_list, prior_list):
            s = series.clamp(min=1e-8)
            p = prior.clamp(min=1e-8)
            kl_sp = (s * (s.log() - p.log())).sum(dim=-1).mean(dim=(1, 2))
            kl_ps = (p * (p.log() - s.log())).sum(dim=-1).mean(dim=(1, 2))
            ad = ad + (kl_sp + kl_ps)
        return ad / len(series_list)

    def compute_loss(self, x):
        recon, s, p = self.forward(x)
        return F.mse_loss(recon, x) - self.lambda_ad * self._association_discrepancy(s, p).mean()

    @torch.no_grad()
    def anomaly_score(self, x):
        recon, s, p = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2)) + self.lambda_ad * self._association_discrepancy(s, p)
