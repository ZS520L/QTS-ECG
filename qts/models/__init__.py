"""Model registry: ``build_model(name, params)`` for QTS and every deep baseline."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from .qts import QuadraticTemplateBank, QuadraticTemplateScorer


def _qts(p):
    return QuadraticTemplateScorer(n_leads=p.get("n_leads", 12),
                                   n_templates=p.get("n_templates", 64),
                                   kernel_sizes=tuple(p.get("kernel_sizes", (155,))),
                                   stride=p.get("stride", 4),
                                   aggregation=p.get("aggregation", "sum"))


def _ae(p):
    from .baselines.ae import AutoEncoder
    return AutoEncoder(in_channels=12, seq_len=5000, channels=p["channels"])


def _vae(p):
    from .baselines.vae import ConvVAE
    return ConvVAE(in_channels=12, seq_len=5000, channels=p["channels"],
                   latent_dim=p["latent_dim"], beta=p["beta"])


def _lstm_ae(p):
    from .baselines.lstm_ae import LSTMAE
    return LSTMAE(in_channels=12, seq_len=5000, hidden_dim=p["hidden_dim"],
                  num_layers=p["num_layers"], downsample=p["downsample"])


def _deep_svdd(p):
    from .baselines.deep_svdd import DeepSVDD
    return DeepSVDD(in_channels=12, seq_len=5000, channels=p["channels"], rep_dim=p["rep_dim"])


def _anomaly_transformer(p):
    from .baselines.anomaly_transformer import AnomalyTransformer
    return AnomalyTransformer(in_channels=12, seq_len=5000, d_model=p["d_model"],
                              n_heads=p["n_heads"], n_layers=p["n_layers"],
                              d_ff=p.get("d_ff", 256), patch_size=p["patch_size"],
                              lambda_ad=p["lambda_ad"])


def _beatgan(p):
    from .baselines.beatgan import BeatGAN
    return BeatGAN(in_channels=12, seq_len=5000, channels=list(p["channels"]),
                   lambda_recon=p["lambda_recon"])


def _dagmm(p):
    from .baselines.dagmm import DAGMM
    return DAGMM(in_channels=12, seq_len=5000, enc_channels=list(p["enc_channels"]),
                 latent_dim=p["latent_dim"], gmm_k=p["gmm_k"])


def _usad(p):
    from .baselines.usad import USAD
    return USAD(in_channels=12, seq_len=5000, channels=list(p["channels"]),
                latent_dim=p["latent_dim"])


def _thoc(p):
    from .baselines.thoc import THOC
    return THOC(in_channels=12, seq_len=5000, hidden_dim=p["hidden_dim"],
                n_layers=p["n_layers"], n_clusters=list(p["n_clusters"]),
                downsample=p["downsample"])


def _tranad(p):
    from .baselines.tranad import TranAD
    return TranAD(in_channels=12, seq_len=5000, d_model=p["d_model"], n_heads=p["n_heads"],
                  n_layers=p["n_layers"], d_ff=p["d_ff"], patch_size=p["patch_size"])


MODEL_REGISTRY: dict[str, Callable[[dict], nn.Module]] = {
    "qts": _qts,
    "ae": _ae,
    "vae": _vae,
    "lstm_ae": _lstm_ae,
    "deep_svdd": _deep_svdd,
    "anomaly_transformer": _anomaly_transformer,
    "beatgan": _beatgan,
    "dagmm": _dagmm,
    "usad": _usad,
    "thoc": _thoc,
    "tranad": _tranad,
}


def build_model(arch: str, params: dict) -> nn.Module:
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture '{arch}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](params or {})


__all__ = ["MODEL_REGISTRY", "QuadraticTemplateBank", "QuadraticTemplateScorer", "build_model"]
