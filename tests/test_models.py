"""Every registered model builds from configs/methods.yaml and has the
parameter count of the configuration evaluated in the paper."""

import pytest
import torch

from qts.config import PROJECT_ROOT, load_config
from qts.models import build_model

CFG = load_config(PROJECT_ROOT / "configs" / "methods.yaml")
DEEP = [m for m, c in CFG["methods"].items() if c["kind"] == "deep"]

EXPECTED_PARAMS = {       # trainable parameters, 12 leads x 5000 samples
    "qts": 119_040,
    "deep_svdd": 91_552,
    "ae": 609_612,
    "vae": 31_459_148,
    "lstm_ae": 411_148,
    "beatgan": 914_093,
    "dagmm": 1_829_892,
    "usad": 914_040,
    "thoc": 304_012,
    "tranad": 4_525_400,
    "anomaly_transformer": 4_613_488,
}


@pytest.mark.parametrize("name", DEEP)
def test_parameter_count(name):
    c = CFG["methods"][name]
    model = build_model(c.get("arch", name), c["params"])
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == EXPECTED_PARAMS[name]


@pytest.mark.parametrize("name", DEEP)
def test_forward_loss_and_score(name):
    torch.manual_seed(0)
    c = CFG["methods"][name]
    model = build_model(c.get("arch", name), c["params"])
    x = torch.randn(4, 12, 5000)
    if hasattr(model, "initialize"):
        loader = [(torch.randn(4, 12, 5000), torch.zeros(4, dtype=torch.long)) for _ in range(5)]
        kwargs = {"n_samples": 4} if name == "qts" else {}
        model.initialize(loader, torch.device("cpu"), **kwargs)
    model.train()
    loss = model.compute_loss(x)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    model.eval()
    s = model.anomaly_score(x)
    assert s.shape == (4,) and torch.isfinite(s).all()
