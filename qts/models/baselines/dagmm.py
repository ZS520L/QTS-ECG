"""DAGMM (Zong et al., ICLR 2018), auto-encoder + Gaussian mixture energy.

Compression network: convolutional encoder -> 10-d latent; estimation network
maps ``[z, relative Euclidean error, cosine similarity]`` to ``K = 4``
mixture memberships. Score: sample energy.

Simplification: the GMM parameters used for scoring are re-estimated from
the scored batch itself rather than stored from training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAGMM(nn.Module):
    """DAGMM adapted for 1D ECG signals.

    Compression: Conv1d AE → low-dim z_c
    Estimation: MLP maps [z_c, cosine_sim, recon_error] → GMM assignments
    """

    def __init__(self, in_channels=12, seq_len=5000,
                 enc_channels=None, latent_dim=10,
                 gmm_k=4, lambda_energy=0.1, lambda_cov=0.005):
        super().__init__()
        if enc_channels is None:
            enc_channels = [32, 64, 128]

        self.latent_dim = latent_dim
        self.gmm_k = gmm_k
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov

        # Encoder
        enc_layers = []
        ch = in_channels
        for c in enc_channels:
            enc_layers += [
                nn.Conv1d(ch, c, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ]
            ch = c
        self.encoder = nn.Sequential(*enc_layers)

        # Compute compressed length
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            enc_out = self.encoder(dummy)
            self._enc_shape = enc_out.shape  # (1, C, L')
            flat_dim = enc_out.numel()

        # Projection to latent
        self.fc_encode = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        # Decoder
        dec_channels = enc_channels[::-1] + [in_channels]
        dec_layers = []
        for i in range(len(dec_channels) - 1):
            if i < len(dec_channels) - 2:
                dec_layers += [
                    nn.ConvTranspose1d(dec_channels[i], dec_channels[i+1],
                                       kernel_size=7, stride=2, padding=3, output_padding=1),
                    nn.BatchNorm1d(dec_channels[i+1]),
                    nn.ReLU(inplace=True),
                ]
            else:
                dec_layers.append(
                    nn.ConvTranspose1d(dec_channels[i], dec_channels[i+1],
                                       kernel_size=7, stride=2, padding=3, output_padding=1)
                )
        self.decoder = nn.Sequential(*dec_layers)
        self.seq_len = seq_len

        # Estimation network: input = z_c (latent_dim) + 2 error features
        est_input_dim = latent_dim + 2
        self.estimation = nn.Sequential(
            nn.Linear(est_input_dim, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, gmm_k),
            nn.Softmax(dim=-1),
        )

    def _encode(self, x):
        enc_out = self.encoder(x)
        z_c = self.fc_encode(enc_out.reshape(x.size(0), -1))
        return z_c, enc_out

    def _decode(self, z_c):
        dec_in = self.fc_decode(z_c).reshape(-1, *self._enc_shape[1:])
        x_recon = self.decoder(dec_in)
        if x_recon.shape[-1] != self.seq_len:
            x_recon = F.interpolate(x_recon, size=self.seq_len, mode="linear", align_corners=False)
        return x_recon

    def _error_features(self, x, x_recon):
        """Compute relative Euclidean distance and cosine similarity."""
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        r_flat = x_recon.reshape(B, -1)

        # Relative Euclidean distance
        eucl = torch.norm(x_flat - r_flat, dim=1) / (torch.norm(x_flat, dim=1) + 1e-8)

        # Cosine similarity
        cos = F.cosine_similarity(x_flat, r_flat, dim=1)

        return torch.stack([eucl, cos], dim=1)  # (B, 2)

    def forward(self, x):
        z_c, _ = self._encode(x)
        x_recon = self._decode(z_c)
        err_feats = self._error_features(x, x_recon)
        z = torch.cat([z_c, err_feats], dim=1)  # (B, latent_dim + 2)
        gamma = self.estimation(z)               # (B, gmm_k)
        return x_recon, z, gamma

    def _gmm_params(self, z, gamma):
        """Compute GMM parameters from soft assignments."""
        B, K = gamma.shape
        D = z.shape[1]

        # Soft counts
        gamma_sum = gamma.sum(dim=0)  # (K,)
        phi = gamma_sum / B           # (K,) mixture weights

        # Means
        mu = (gamma.unsqueeze(2) * z.unsqueeze(1)).sum(dim=0) / (gamma_sum.unsqueeze(1) + 1e-8)  # (K, D)

        # Covariances
        z_centered = z.unsqueeze(1) - mu.unsqueeze(0)  # (B, K, D)
        cov = torch.zeros(K, D, D, device=z.device)
        for k in range(K):
            diff = z_centered[:, k, :]  # (B, D)
            weighted = diff * gamma[:, k].unsqueeze(1)  # (B, D)
            cov[k] = (weighted.t() @ diff) / (gamma_sum[k] + 1e-8)
            cov[k] += 1e-4 * torch.eye(D, device=z.device)  # regularize

        return phi, mu, cov

    def _gmm_energy(self, z, phi, mu, cov):
        """Compute sample energy (negative log-likelihood under GMM)."""
        K, D = mu.shape
        energy = []
        for k in range(K):
            diff = z - mu[k].unsqueeze(0)  # (B, D)
            cov_inv = torch.linalg.inv(cov[k])
            log_det = torch.logdet(cov[k])
            mahal = (diff @ cov_inv * diff).sum(dim=1)  # (B,)
            log_prob = -0.5 * (D * torch.log(torch.tensor(2 * 3.14159, device=z.device))
                               + log_det + mahal)
            energy.append(torch.log(phi[k] + 1e-8) + log_prob)
        energy = torch.stack(energy, dim=1)  # (B, K)
        return -torch.logsumexp(energy, dim=1)  # (B,)

    def compute_loss(self, x):
        x_recon, z, gamma = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)

        phi, mu, cov = self._gmm_params(z, gamma)
        energy = self._gmm_energy(z, phi, mu, cov)
        energy_loss = energy.mean()

        # Diagonal penalty
        diag_penalty = sum(1.0 / (torch.diagonal(cov[k]) + 1e-8).sum()
                          for k in range(self.gmm_k))

        return recon_loss + self.lambda_energy * energy_loss + self.lambda_cov * diag_penalty

    @torch.no_grad()
    def anomaly_score(self, x):
        x_recon, z, gamma = self.forward(x)
        phi, mu, cov = self._gmm_params(z, gamma)
        return self._gmm_energy(z, phi, mu, cov)
