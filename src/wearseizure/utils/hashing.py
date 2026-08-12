"""Canonical hashing for manifests, splits, and frozen postprocess params.

Every artifact that gates a downstream stage (manifest -> splits -> frozen
thresholds) carries a hash of its own content plus the hash of whatever it was
derived from, so a stale split against a regenerated manifest is a hard error
rather than a silent bug.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _default(o: Any) -> Any:
    # numpy scalars (int64, float64, ...) implement .item() to convert to the
    # equivalent native Python type -- without this, they would fall through
    # to str(o) and hash differently from the native int/float produced
    # before a DataFrame round-trip through CSV (pandas dtypes vs. plain
    # Python types), even though the value is identical.
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
