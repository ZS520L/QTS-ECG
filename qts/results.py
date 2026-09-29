"""Helpers to collect per-run ``metrics.json`` files into tidy DataFrames."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_metrics(root: str | Path) -> pd.DataFrame:
    """One row per ``<root>/<dataset>/<method>/seed_<s>/metrics.json``."""
    rows = []
    for p in sorted(Path(root).glob("*/*/seed_*/metrics.json")):
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        m.setdefault("dataset", p.parts[-4])
        m.setdefault("method", p.parts[-3])
        m.setdefault("seed", int(p.parent.name.split("_")[1]))
        m.pop("scale_weights", None)
        rows.append(m)
    return pd.DataFrame(rows)


def collect_errors(root: str | Path) -> list[str]:
    return [str(p.parent) for p in sorted(Path(root).glob("*/*/seed_*/error.txt"))]
