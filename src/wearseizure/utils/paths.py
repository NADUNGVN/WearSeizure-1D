"""Path resolution helpers so no absolute machine-specific path is hardcoded.

Configs reference paths via Hydra profile groups (``configs/profile/*.yaml``),
which in turn interpolate environment variables for the server profile. This
module only centralizes the small amount of Python-side path logic (e.g.
resolving relative to the repo root) that configs cannot express on their own.
"""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
