"""Threshold selection frozen on validation only (memo 5.1 step 3 / 5.4).

Nothing in this module -- or anything downstream of `FrozenPostprocessParams`
-- is allowed to see continuous test data while a threshold is being chosen.
The frozen params are hashed and serializable so a test run can prove which
exact (on, off, run_length, gap) it used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from wearseizure.eval.metrics_event import compute_event_metrics
from wearseizure.postprocess.hysteresis import PostprocessParams
from wearseizure.postprocess.pipeline import run_postprocess
from wearseizure.utils.hashing import sha256_of


@dataclass(frozen=True)
class FrozenPostprocessParams:
    params: PostprocessParams
    val_sensitivity: float
    val_far_per_hour: float
    fold_id: str
    params_hash: str

    def to_dict(self) -> dict:
        return {
            "params": asdict(self.params),
            "val_sensitivity": self.val_sensitivity,
            "val_far_per_hour": self.val_far_per_hour,
            "fold_id": self.fold_id,
            "params_hash": self.params_hash,
        }

    @staticmethod
    def from_dict(d: dict) -> FrozenPostprocessParams:
        return FrozenPostprocessParams(
            params=PostprocessParams(**d["params"]),
            val_sensitivity=d["val_sensitivity"],
            val_far_per_hour=d["val_far_per_hour"],
            fold_id=d["fold_id"],
            params_hash=d["params_hash"],
        )


def fit_threshold_on_val(
    end_sec_by_edf: dict[str, np.ndarray],
    scores_by_edf: dict[str, np.ndarray],
    events_by_edf: dict[str, list[tuple[str, float, float]]],
    exposure_hours_by_edf: dict[str, float],
    method: str,
    ema_alpha: float,
    run_length: int,
    event_merge_gap_s: float,
    threshold_on_grid: list[float],
    threshold_off_grid: list[float],
    fold_id: str,
    far_weight: float = 0.1,
) -> FrozenPostprocessParams:
    """Grid-search (threshold_on, threshold_off) to maximize
    `sensitivity - far_weight * far_per_hour` pooled across all val EDFs.
    `far_weight` is a tunable heuristic, not a memo-specified constant --
    adjust it once real data is available if the tradeoff needs shifting.

    Only supports `method="hysteresis_runlength"` (the primary postprocessor,
    memo 4.5). The `raw_threshold`/`ema` ablation variants (memo 7.2) use a
    single `threshold`, not an (on, off) pair, and are compared manually
    during ablation rather than through this automatic search.
    """
    if method != "hysteresis_runlength":
        raise NotImplementedError(
            f"fit_threshold_on_val only supports method='hysteresis_runlength', got {method!r}"
        )
    best: tuple[PostprocessParams, float, float] | None = None
    best_score = -np.inf

    for on in threshold_on_grid:
        for off in threshold_off_grid:
            if off >= on:
                continue
            params = PostprocessParams(
                method=method, ema_alpha=ema_alpha, threshold_on=on, threshold_off=off,
                run_length=run_length, event_merge_gap_s=event_merge_gap_s,
            )
            total_events = total_matched = total_false_alarms = 0
            total_exposure = 0.0
            for edf_id, end_sec in end_sec_by_edf.items():
                alarms = run_postprocess(end_sec, scores_by_edf[edf_id], params)
                m = compute_event_metrics(events_by_edf.get(edf_id, []), alarms, exposure_hours_by_edf[edf_id])
                total_events += m.n_events
                total_matched += m.n_matched
                total_false_alarms += m.n_false_alarms
                total_exposure += m.exposure_hours

            sensitivity = total_matched / total_events if total_events else 0.0
            far = total_false_alarms / total_exposure if total_exposure else 0.0
            objective = sensitivity - far_weight * far
            if objective > best_score:
                best_score = objective
                best = (params, sensitivity, far)

    if best is None:
        raise RuntimeError(
            "threshold search produced no valid (on, off) combination "
            f"(grids: on={threshold_on_grid}, off={threshold_off_grid})"
        )

    params, sensitivity, far = best
    return FrozenPostprocessParams(
        params=params,
        val_sensitivity=sensitivity,
        val_far_per_hour=far,
        fold_id=fold_id,
        params_hash=sha256_of(asdict(params)),
    )


def save_frozen_params(frozen: FrozenPostprocessParams, path: str) -> None:
    Path(path).write_text(json.dumps(frozen.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_frozen_params(path: str) -> FrozenPostprocessParams:
    return FrozenPostprocessParams.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
