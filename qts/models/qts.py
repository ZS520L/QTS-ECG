"""Quadratic Template Scoring (QTS).

A bank of ``M`` learnable templates ``w_k`` of shape ``(C, K)`` is slid over
the input ``x`` of shape ``(C, L)`` with stride ``s``::

    d_{k,t} = sum_{c,i} (x_{c, st+i} + w_{k,c,i})^2     squared mismatch
    s_t     = min_k d_{k,t}                             best template
    S(x)    = mean_t s_t                                anomaly score

Templates are stored with a negative sign (``x`` matches template ``k`` when
``x = -w_k``) so that the cross term is a plain convolution. Expanding the
square gives::

    d_{k,t} = ||x_t||^2 + 2 <w_k, x_t> + ||w_k||^2

i.e. one convolution with an all-ones kernel (window energy), one ordinary
convolution with the template bank, and a per-template constant. The model
has no non-linearity other than ``min`` and is trained by minimising
``S(x)`` on normal recordings only.

The paper's configuration is a single bank with ``K = 155`` samples (310 ms
at 500 Hz, the QT interval), ``M = 64`` and stride 4, i.e. 119,040
parameters. Multi-scale variants (used in the ablation) sum or otherwise
aggregate the scores of several banks.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadraticTemplateBank(nn.Module):
    """One template bank: ``M`` templates of shape ``(n_leads, kernel_size)``."""

    def __init__(self, n_leads: int = 12, n_templates: int = 64,
                 kernel_size: int = 155, stride: int = 4):
        super().__init__()
        self.n_leads = n_leads
        self.n_templates = n_templates
        self.kernel_size = kernel_size
        self.stride = stride
        self.templates = nn.Parameter(torch.randn(n_templates, n_leads, kernel_size) * 0.1)
        self.register_buffer("ones_kernel", torch.ones(1, n_leads, kernel_size),
                             persistent=False)

    @torch.no_grad()
    def init_from_data(self, x: torch.Tensor, jitter: float = 0.01) -> None:
        """Set each template to the negative of a random window of real data.

        Analogous to k-means++ seeding: one random recording and one random
        offset per template, plus Gaussian jitter to break ties.
        """
        n, c, length = x.shape
        assert c == self.n_leads and length >= self.kernel_size
        idx_n = torch.randint(0, n, (self.n_templates,))
        idx_t = torch.randint(0, length - self.kernel_size + 1, (self.n_templates,))
        w = torch.empty_like(self.templates)
        for k, (i, t) in enumerate(zip(idx_n.tolist(), idx_t.tolist())):
            w[k] = -x[i, :, t:t + self.kernel_size]
        w += jitter * torch.randn_like(w)
        self.templates.data.copy_(w)

    def distance_map(self, x: torch.Tensor) -> torch.Tensor:
        """Squared distances ``(B, M, T)`` between every window and template."""
        energy = F.conv1d(x * x, self.ones_kernel, stride=self.stride)       # (B, 1, T)
        cross = F.conv1d(x, self.templates, stride=self.stride)               # (B, M, T)
        w_norm = (self.templates * self.templates).sum(dim=(1, 2))            # (M,)
        d = energy + 2.0 * cross + w_norm.view(1, -1, 1)
        return d.clamp_min_(0.0)  # guard against tiny negative round-off

    def best_match(self, x: torch.Tensor):
        """Per-window best distance and template index, each ``(B, T)``."""
        return self.distance_map(x).min(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        best, _ = self.best_match(x)
        return best.mean(dim=1)


class QuadraticTemplateScorer(nn.Module):
    """QTS with one or more template banks.

    Args:
        n_leads: number of ECG leads.
        n_templates: templates per bank (int or one value per bank).
        kernel_sizes: template length of each bank, in samples.
        stride: sliding-window stride (int or one value per bank).
        aggregation: how bank scores are combined - ``"sum"`` (default),
            ``"learned"`` (softmax-normalised learnable weights) or ``"max"``.
    """

    def __init__(self, n_leads: int = 12, n_templates: int | Sequence[int] = 64,
                 kernel_sizes: Sequence[int] = (155,), stride: int | Sequence[int] = 4,
                 aggregation: str = "sum"):
        super().__init__()
        n = len(kernel_sizes)
        n_templates = [n_templates] * n if isinstance(n_templates, int) else list(n_templates)
        stride = [stride] * n if isinstance(stride, int) else list(stride)
        assert len(n_templates) == len(stride) == n
        self.banks = nn.ModuleList([
            QuadraticTemplateBank(n_leads, n_templates[i], kernel_sizes[i], stride[i])
            for i in range(n)
        ])
        self.kernel_sizes = tuple(kernel_sizes)
        if aggregation not in ("sum", "learned", "max"):
            raise ValueError(aggregation)
        self.aggregation = aggregation
        if aggregation == "learned":
            self.weight_logits = nn.Parameter(torch.zeros(n))

    # -- training hooks used by qts.engine ---------------------------------
    @torch.no_grad()
    def initialize(self, loader, device, n_samples: int = 256) -> None:
        """Data-driven initialisation from the first ``n_samples`` training records."""
        batches, seen = [], 0
        for x, _ in loader:
            batches.append(x)
            seen += x.shape[0]
            if seen >= n_samples:
                break
        x = torch.cat(batches, 0)[:n_samples].to(device)
        for bank in self.banks:
            bank.init_from_data(x)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean()

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # -----------------------------------------------------------------------
    def per_bank_scores(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [bank(x) for bank in self.banks]

    def scale_weights(self) -> list[float]:
        if self.aggregation == "learned":
            return torch.softmax(self.weight_logits, 0).detach().cpu().tolist()
        return [1.0] * len(self.banks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.banks) == 1:
            return self.banks[0](x)
        scores = self.per_bank_scores(x)
        if self.aggregation == "max":
            return torch.stack(scores, 0).max(dim=0).values
        if self.aggregation == "learned":
            w = torch.softmax(self.weight_logits, 0)
            return sum(wi * si for wi, si in zip(w, scores))
        return sum(scores)
