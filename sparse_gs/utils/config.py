"""Configuration utilities.

YAML-based configs with a lightweight ``_base_`` inheritance mechanism
(deep-merge child onto parent). The resulting config is a plain ``dict`` —
we deliberately avoid OmegaConf / Hydra to keep the dep surface minimal.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config file, resolving a single ``_base_`` reference.

    ``_base_`` is interpreted relative to the config file that contains it.
    """
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref is not None:
        base_path = (path.parent / base_ref).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def save_config(cfg: Dict[str, Any], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """``cfg['a']['b']['c']`` <=> ``get(cfg, 'a.b.c')``."""
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
