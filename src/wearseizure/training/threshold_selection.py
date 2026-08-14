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
    far_cap_per_hour: float | None = None,
) -> FrozenPostprocessParams:
    """Grid-search (threshold_on, threshold_off) over all val EDFs.

    Two selection policies:

    - `far_cap_per_hour` given (preferred -- matches memo Table 6's separate,
      independent gates rather than blending them): among combos whose
      validation FAR is <= the cap, pick the one with the highest
      sensitivity (ties broken by lower FAR). If no combo satisfies the cap,
      fall back to the single lowest-FAR combo. This is what
      `configs/postprocess/hysteresis_runlength.yaml`'s `far_cap_per_hour`
      sets by default.
    - `far_cap_per_hour=None`: maximize the linear blend
      `sensitivity - far_weight * far_per_hour`. Kept for cases with no
      explicit FAR target to aim for; in practice this tends to either
      under-shoot sensitivity (if it favors low FAR) or over-shoot FAR (if
      widened thresholds are available) since it has no hard constraint --
      the capped policy above is more predictable once a target is known.

    Only supports `method="hysteresis_runlength"` (the primary postprocessor,
    memo 4.5). The `raw_threshold`/`ema` ablation variants (memo 7.2) use a
    single `threshold`, not an (on, off) pair, and are compared manually
    during ablation rather than through this automatic search.
    """
    if method != "hysteresis_runlength":
        raise NotImplementedError(
            f"fit_threshold_on_val only supports method='hysteresis_runlength', got {method!r}"
        )

    candidates: list[tuple[PostprocessParams, float, float]] = []
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
            candidates.append((params, sensitivity, far))

    if not candidates:
        raise RuntimeError(
            "threshold search produced no valid (on, off) combination "
            f"(grids: on={threshold_on_grid}, off={threshold_off_grid})"
        )

    if far_cap_per_hour is not None:
        passing = [c for c in candidates if c[2] <= far_cap_per_hour]
        best = max(passing, key=lambda c: (c[1], -c[2])) if passing else min(candidates, key=lambda c: c[2])
    else:
        best = max(candidates, key=lambda c: c[1] - far_weight * c[2])

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
