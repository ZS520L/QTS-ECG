"""THOC (Shen et al., NeurIPS 2020), temporal hierarchical one-class network.

Input down-sampled by 10 -> linear projection -> 3 stacked GRUs with
2x temporal pooling between levels -> time-averaged representation at each
level -> distance to the nearest of ``[4, 8, 16]`` learnable centroids.
Score: sum of nearest-centroid distances over levels.

Simplification: plain GRUs with pooling instead of dilated RNNs, and no
self-supervised auxiliary losses. Centroids are initialised from random
training representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class THOC(nn.Module):
    """Temporal Hierarchical One-Class Network for ECG.

    Input: (B, 12, 5000) → downsample → GRU layers → hierarchical centroids.
    """

    def __init__(self, in_channels=12, seq_len=5000,
                 hidden_dim=128, n_layers=3,
                 n_clusters=None, downsample=10):
        super().__init__()
        if n_clusters is None:
            n_clusters = [4, 8, 16]
        assert len(n_clusters) == n_layers

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.downsample = downsample
        self.ds_len = seq_len // downsample  # 500

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # GRU layers
        self.gru_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gru_layers.append(
                nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            )

        # Temporal pooling between layers (reduce sequence length by 2)
        self.pool_factor = 2

        # Cluster centroids at each level
        self.centroids = nn.ParameterList()
        for i, k in enumerate(n_clusters):
            self.centroids.append(
                nn.Parameter(torch.randn(k, hidden_dim) * 0.01)
            )

        # Output projection for reconstruction
        self.output_proj = nn.Linear(hidden_dim, in_channels)

    def _hierarchical_forward(self, x):
        """Forward through hierarchical GRU layers.

        Returns list of representation tensors at each level.
        """
        # x: (B, T, hidden_dim)
        reps = []
        h = x
        for i, gru in enumerate(self.gru_layers):
            h, _ = gru(h)  # (B, T_i, hidden_dim)
            reps.append(h)  # Store representations at this level
            # Temporal pooling for next level
            if i < self.n_layers - 1:
                T = h.shape[1]
                if T > 1:
                    # Average pool to reduce temporal dim
                    T_new = max(T // self.pool_factor, 1)
                    h = h[:, :T_new * self.pool_factor].reshape(
                        h.shape[0], T_new, self.pool_factor, -1).mean(dim=2)
        return reps

    def _centroid_distances(self, reps):
        """Compute min distance to centroids at each level."""
        distances = []
        for i, (rep, centroids) in enumerate(zip(reps, self.centroids)):
            # rep: (B, T_i, D), centroids: (K, D)
            # Mean-pool temporal dim to get single representation per sample
            rep_mean = rep.mean(dim=1)  # (B, D)
            # Distance to each centroid
            dists = torch.cdist(rep_mean.unsqueeze(1),
                              centroids.unsqueeze(0))  # (B, 1, K)
            min_dist = dists.squeeze(1).min(dim=1).values  # (B,)
            distances.append(min_dist)
        return distances

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 12, 5000)
        Returns:
            reps: list of hierarchical representations
            distances: list of centroid distances at each level
        """
        B = x.shape[0]
        # Downsample + transpose: (B, 12, 5000) → (B, 500, 12)
        x_ds = x[:, :, ::self.downsample].transpose(1, 2)  # (B, ds_len, 12)
        h = self.input_proj(x_ds)  # (B, ds_len, hidden_dim)

        reps = self._hierarchical_forward(h)
        distances = self._centroid_distances(reps)

        return reps, distances

    def compute_loss(self, x):
        """SVDD-style loss: minimize distance to nearest centroid at all levels."""
        reps, distances = self.forward(x)
        total_loss = sum(d.mean() for d in distances)
        return total_loss

    @torch.no_grad()
    def anomaly_score(self, x):
        """Sum of min-centroid distances across all levels."""
        _, distances = self.forward(x)
        score = sum(d for d in distances)  # (B,)
        return score

    def initialize(self, loader, device):
        self.init_centroids(loader, device)

    def init_centroids(self, loader, device):
        """Initialize centroids using k-means on training data representations."""
        self.eval()
        all_reps = [[] for _ in range(self.n_layers)]
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                reps, _ = self.forward(xb)
                for i, rep in enumerate(reps):
                    all_reps[i].append(rep.mean(dim=1).cpu())  # (B, D)

        for i in range(self.n_layers):
            all_rep = torch.cat(all_reps[i], dim=0)  # (N, D)
            K = self.centroids[i].shape[0]
            # Simple k-means++ init: random subset
            idx = torch.randperm(all_rep.shape[0])[:K]
            self.centroids[i].data.copy_(all_rep[idx].to(device))
        self.train()
