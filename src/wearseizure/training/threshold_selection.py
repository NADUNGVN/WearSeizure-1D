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


ValFoldData = tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, list[tuple[str, float, float]]],
    dict[str, float],
]  # (end_sec_by_edf, scores_by_edf, events_by_edf, exposure_hours_by_edf)


def _score_candidate_totals(
    end_sec_by_edf: dict[str, np.ndarray],
    scores_by_edf: dict[str, np.ndarray],
    events_by_edf: dict[str, list[tuple[str, float, float]]],
    exposure_hours_by_edf: dict[str, float],
    params: PostprocessParams,
) -> tuple[int, int, int, float]:
    """(n_events, n_matched, n_false_alarms, exposure_hours) for one candidate
    `params` against one fold's (or, for pooling, one EDF-group's) val data.
    """
    total_events = total_matched = total_false_alarms = 0
    total_exposure = 0.0
    for edf_id, end_sec in end_sec_by_edf.items():
        alarms = run_postprocess(end_sec, scores_by_edf[edf_id], params)
        m = compute_event_metrics(events_by_edf.get(edf_id, []), alarms, exposure_hours_by_edf[edf_id])
        total_events += m.n_events
        total_matched += m.n_matched
        total_false_alarms += m.n_false_alarms
        total_exposure += m.exposure_hours
    return total_events, total_matched, total_false_alarms, total_exposure


def _select_best(
    candidates: list[tuple[PostprocessParams, float, float]],
    far_weight: float,
    far_cap_per_hour: float | None,
) -> tuple[PostprocessParams, float, float]:
    if far_cap_per_hour is not None:
        passing = [c for c in candidates if c[2] <= far_cap_per_hour]
        return max(passing, key=lambda c: (c[1], -c[2])) if passing else min(candidates, key=lambda c: c[2])
    return max(candidates, key=lambda c: c[1] - far_weight * c[2])


def fit_threshold_on_val_pooled(
    val_folds: list[ValFoldData],
    method: str,
    ema_alpha: float,
    run_length: int,
    event_merge_gap_s: float,
    threshold_on_grid: list[float],
    threshold_off_grid: list[float],
    group_id: str,
    far_weight: float = 0.1,
    far_cap_per_hour: float | None = None,
) -> FrozenPostprocessParams:
    """Like `fit_threshold_on_val`, but pools validation evidence across
    multiple folds' (end_sec, scores, events, exposure) sets -- each already
    scored by *that fold's own* trained model -- into one shared threshold.

    Motivation (memo 5.1/5.3): a patient with only 1-2 seizures total gives
    each of their personalized folds a validation set with just 1 event,
    which is too noisy to pick a reliable threshold from alone (see
    docs/SERVER_INVENTORY.md failure analysis: several such patients hit
    val_sensitivity=0.0 at the loosest available threshold). Pooling that
    patient's several folds' val evidence together (still never touching
    test data) gives the search more than 1-2 data points to work with,
    without changing which model scores which EDF.

    `group_id` is typically a patient/subject id, not a single fold_id.
    """
    if method != "hysteresis_runlength":
        raise NotImplementedError(
            f"fit_threshold_on_val_pooled only supports method='hysteresis_runlength', got {method!r}"
        )
    if not val_folds:
        raise ValueError(f"fit_threshold_on_val_pooled: no val folds given for group {group_id!r}")

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
            for end_sec_by_edf, scores_by_edf, events_by_edf, exposure_hours_by_edf in val_folds:
                events, matched, false_alarms, exposure = _score_candidate_totals(
                    end_sec_by_edf, scores_by_edf, events_by_edf, exposure_hours_by_edf, params
                )
                total_events += events
                total_matched += matched
                total_false_alarms += false_alarms
                total_exposure += exposure

            sensitivity = total_matched / total_events if total_events else 0.0
            far = total_false_alarms / total_exposure if total_exposure else 0.0
            candidates.append((params, sensitivity, far))

    if not candidates:
        raise RuntimeError(
            "threshold search produced no valid (on, off) combination "
            f"(grids: on={threshold_on_grid}, off={threshold_off_grid})"
        )

    params, sensitivity, far = _select_best(candidates, far_weight, far_cap_per_hour)
    return FrozenPostprocessParams(
        params=params,
        val_sensitivity=sensitivity,
        val_far_per_hour=far,
        fold_id=group_id,
        params_hash=sha256_of(asdict(params)),
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
    """Grid-search (threshold_on, threshold_off) over all val EDFs of a single
    fold. Thin wrapper around `fit_threshold_on_val_pooled` with one fold's
    data -- see that function for the two selection policies (far_cap_per_hour
    vs. the far_weight blend) and why `far_cap_per_hour` is preferred.

    Only supports `method="hysteresis_runlength"` (the primary postprocessor,
    memo 4.5). The `raw_threshold`/`ema` ablation variants (memo 7.2) use a
    single `threshold`, not an (on, off) pair, and are compared manually
    during ablation rather than through this automatic search.
    """
    return fit_threshold_on_val_pooled(
        val_folds=[(end_sec_by_edf, scores_by_edf, events_by_edf, exposure_hours_by_edf)],
        method=method, ema_alpha=ema_alpha, run_length=run_length, event_merge_gap_s=event_merge_gap_s,
        threshold_on_grid=threshold_on_grid, threshold_off_grid=threshold_off_grid,
        group_id=fold_id, far_weight=far_weight, far_cap_per_hour=far_cap_per_hour,
    )


def save_frozen_params(frozen: FrozenPostprocessParams, path: str) -> None:
    Path(path).write_text(json.dumps(frozen.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_frozen_params(path: str) -> FrozenPostprocessParams:
    return FrozenPostprocessParams.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
