"""YAML configuration loading.

Configurations are plain nested dictionaries. ``load_config`` supports a
single level of inheritance through a top-level ``defaults`` key that lists
other YAML files (relative to the including file) to be merged first.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict, other: dict) -> dict:
    """Recursively merge ``other`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    merged: dict[str, Any] = {}
    for parent in cfg.pop("defaults", []) or []:
        merged = deep_update(merged, load_config(path.parent / parent))
    return deep_update(merged, cfg)


def resolve_path(p: str | os.PathLike) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else (PROJECT_ROOT / p)
