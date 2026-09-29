import torch

from qts.models.qts import QuadraticTemplateBank, QuadraticTemplateScorer


def naive_distance_map(bank: QuadraticTemplateBank, x: torch.Tensor) -> torch.Tensor:
    """Direct evaluation of sum_{c,i} (x_{c,st+i} + w_{k,c,i})^2 via unfold."""
    win = x.unfold(2, bank.kernel_size, bank.stride)            # (B, C, T, K)
    win = win.permute(0, 2, 1, 3).unsqueeze(1)                   # (B, 1, T, C, K)
    w = bank.templates.unsqueeze(0).unsqueeze(2)                 # (1, M, 1, C, K)
    return ((win + w) ** 2).sum(dim=(-1, -2))                    # (B, M, T)


def test_convolutional_expansion_matches_naive_definition():
    torch.manual_seed(0)
    bank = QuadraticTemplateBank(n_leads=3, n_templates=5, kernel_size=11, stride=3).double()
    x = torch.randn(2, 3, 100, dtype=torch.float64)
    fast = bank.distance_map(x)
    slow = naive_distance_map(bank, x)
    assert fast.shape == slow.shape
    assert torch.allclose(fast, slow, atol=1e-9)
    assert torch.allclose(bank(x), slow.min(dim=1).values.mean(dim=1), atol=1e-9)


def test_paper_configuration_parameter_count():
    model = QuadraticTemplateScorer(n_leads=12, n_templates=64, kernel_sizes=(155,), stride=4)
    assert sum(p.numel() for p in model.parameters()) == 64 * 12 * 155 == 119_040


def test_data_initialisation_places_templates_on_negated_windows():
    torch.manual_seed(0)
    bank = QuadraticTemplateBank(n_leads=2, n_templates=4, kernel_size=7, stride=1)
    x = torch.randn(3, 2, 50)
    bank.init_from_data(x, jitter=0.0)
    # every template's best match somewhere in the data is (numerically) zero
    d = bank.distance_map(x).amin(dim=(0, 2))
    assert torch.all(d < 1e-4)


def test_score_is_zero_for_a_signal_made_of_templates():
    bank = QuadraticTemplateBank(n_leads=1, n_templates=1, kernel_size=5, stride=5)
    motif = torch.randn(1, 5)
    bank.templates.data = -motif.unsqueeze(0)
    x = motif.repeat(1, 4).unsqueeze(0)                          # 4 exact repetitions
    assert bank(x).item() < 1e-6


def test_multiscale_aggregations():
    torch.manual_seed(0)
    x = torch.randn(2, 12, 800)
    for agg in ("sum", "learned", "max"):
        m = QuadraticTemplateScorer(12, 8, kernel_sizes=(35, 155), stride=4, aggregation=agg)
        s = m.anomaly_score(x)
        parts = torch.stack(m.per_bank_scores(x))
        assert s.shape == (2,)
        if agg == "sum":
            assert torch.allclose(s, parts.sum(0))
        if agg == "max":
            assert torch.allclose(s, parts.max(0).values)
        if agg == "learned":
            assert torch.allclose(s, parts.mean(0), atol=1e-5)  # equal weights at init
