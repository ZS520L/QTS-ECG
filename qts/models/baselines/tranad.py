"""TranAD (Tuli et al., VLDB 2022), transformer reconstruction baseline.

The recording is cut into 100 patches of 50 samples x 12 leads; a transformer
encoder and two transformer decoders reconstruct the patches. Score: mean of
the two decoders' reconstruction errors.

Simplification: the two-phase self-conditioning / adversarial training of
the original is replaced by the sum of both reconstruction losses plus a
consistency term between the decoders.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TranAD(nn.Module):
    """TranAD for ECG anomaly detection.

    Input: (B, 12, 5000) → patching → transformer encode/decode → reconstruct.
    """

    def __init__(self, in_channels=12, seq_len=5000,
                 patch_size=50, d_model=256, n_heads=4,
                 n_layers=2, d_ff=512):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.seq_len = seq_len
        self.d_model = d_model

        # Patch embedding
        patch_dim = in_channels * patch_size  # 12 * 50 = 600
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=self.n_patches + 10)

        # Shared encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Decoder 1
        dec_layer1 = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, dropout=0.1)
        self.decoder1 = nn.TransformerDecoder(dec_layer1, num_layers=n_layers)

        # Decoder 2
        dec_layer2 = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, dropout=0.1)
        self.decoder2 = nn.TransformerDecoder(dec_layer2, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, patch_dim)

        self._step = 0

    def _patchify(self, x):
        """(B, 12, 5000) → (B, n_patches, patch_dim)"""
        B = x.shape[0]
        # Reshape to patches
        x = x[:, :, :self.n_patches * self.patch_size]
        x = x.reshape(B, x.shape[1], self.n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)
        return x

    def _unpatchify(self, patches, in_channels=12):
        """(B, n_patches, patch_dim) → (B, 12, seq_len)"""
        B = patches.shape[0]
        patches = patches.reshape(B, self.n_patches, in_channels, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B, in_channels, -1)
        if patches.shape[-1] != self.seq_len:
            patches = F.interpolate(patches, size=self.seq_len,
                                   mode="linear", align_corners=False)
        return patches

    def forward(self, x):
        """Returns recon1, recon2 (from two decoders)."""
        patches = self._patchify(x)  # (B, n_patches, patch_dim)
        emb = self.pos_enc(self.patch_embed(patches))  # (B, n_patches, d_model)

        # Encode
        memory = self.encoder(emb)

        # Decode with both decoders
        out1 = self.decoder1(emb, memory)
        out2 = self.decoder2(emb, memory)

        # Project back to patch space
        recon1 = self._unpatchify(self.output_proj(out1))
        recon2 = self._unpatchify(self.output_proj(out2))

        return recon1, recon2

    def compute_loss(self, x):
        """TranAD adversarial loss.

        Simplified: both decoders minimize recon error,
        with focus score from their disagreement.
        """
        recon1, recon2 = self.forward(x)
        loss1 = F.mse_loss(recon1, x)
        loss2 = F.mse_loss(recon2, x)

        self._step += 1
        # Adversarial: decoder2 also tries to reconstruct decoder1's errors
        focus = F.mse_loss(recon1.detach(), recon2)

        return loss1 + loss2 + 0.5 * focus

    @torch.no_grad()
    def anomaly_score(self, x):
        recon1, recon2 = self.forward(x)
        score1 = ((x - recon1) ** 2).mean(dim=(1, 2))
        score2 = ((x - recon2) ** 2).mean(dim=(1, 2))
        return (score1 + score2) / 2
