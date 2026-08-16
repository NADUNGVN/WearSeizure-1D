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
    # Validation-set mean detection delay of the chosen combination. Recorded
    # even under the default objective so the cost of a threshold choice is
    # visible on the axis that never cleared its gate.
    val_delay_mean_s: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "params": asdict(self.params),
            "val_sensitivity": self.val_sensitivity,
            "val_far_per_hour": self.val_far_per_hour,
            "val_delay_mean_s": self.val_delay_mean_s,
            "fold_id": self.fold_id,
            "params_hash": self.params_hash,
        }

    @staticmethod
    def from_dict(d: dict) -> FrozenPostprocessParams:
        return FrozenPostprocessParams(
            params=PostprocessParams(**d["params"]),
            val_sensitivity=d["val_sensitivity"],
            val_far_per_hour=d["val_far_per_hour"],
            # Absent in files frozen before the delay axis was recorded.
            val_delay_mean_s=d.get("val_delay_mean_s", float("nan")),
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
) -> tuple[int, int, int, float, list[float]]:
    """(n_events, n_matched, n_false_alarms, exposure_hours, delays_s) for one
    candidate `params` against one fold's (or, for pooling, one EDF-group's)
    val data.
    """
    total_events = total_matched = total_false_alarms = 0
    total_exposure = 0.0
    delays: list[float] = []
    for edf_id, end_sec in end_sec_by_edf.items():
        alarms = run_postprocess(end_sec, scores_by_edf[edf_id], params)
        m = compute_event_metrics(events_by_edf.get(edf_id, []), alarms, exposure_hours_by_edf[edf_id])
        total_events += m.n_events
        total_matched += m.n_matched
        total_false_alarms += m.n_false_alarms
        total_exposure += m.exposure_hours
        delays.extend(m.delays_s)
    return total_events, total_matched, total_false_alarms, total_exposure, delays


Candidate = tuple[PostprocessParams, float, float, float]  # (params, sensitivity, far, mean_delay_s)

OBJECTIVES = ("max_sensitivity", "min_delay")


def _select_best(
    candidates: list[Candidate],
    far_weight: float,
    far_cap_per_hour: float | None,
    objective: str = "max_sensitivity",
    sensitivity_floor: float | None = None,
) -> Candidate:
    """Pick one candidate threshold pair.

    `max_sensitivity` (default) is the original policy and is unchanged: under a
    FAR cap, take the highest-sensitivity combination; with no cap, take the
    best `sensitivity - far_weight * far` blend.

    `min_delay` exists because the 19 real-data runs in
    docs/EXPERIMENT_LOG_G1a.md show the project spending effort on the axis it
    already wins by a wide margin. The best FAR reached is 0.0621/h against a
    <=0.30/h gate -- roughly 5x more headroom than required -- while
    detection delay was never once inside its gate. Run #19 is the clearest
    case: it bought FAR 0.0621/h at the cost of sensitivity (0.876 -> 0.824),
    worst-patient sensitivity (0.25 -> 0.00) and delay (23.5s -> 26.9s).

    Under `min_delay` the surplus is spent instead of hoarded: among the
    combinations that stay inside the FAR cap *and* hold sensitivity at or
    above `sensitivity_floor`, take the fastest. Ties break toward higher
    sensitivity, then lower FAR. If nothing clears both constraints the
    sensitivity floor is dropped first (FAR is the clinical constraint, so it
    is relaxed last), and only then does it fall back to the minimum-FAR
    candidate, which is what the original policy also does.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}, expected one of {OBJECTIVES}")

    if objective == "min_delay":
        within_far = [c for c in candidates if far_cap_per_hour is None or c[2] <= far_cap_per_hour]
        if sensitivity_floor is not None:
            within_both = [c for c in within_far if c[1] >= sensitivity_floor]
            if within_both:
                return min(within_both, key=lambda c: (c[3], -c[1], c[2]))
        if within_far:
            return min(within_far, key=lambda c: (c[3], -c[1], c[2]))
        return min(candidates, key=lambda c: c[2])

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
    objective: str = "max_sensitivity",
    sensitivity_floor: float | None = None,
    window_s: float = 0.0,
    alarm_timestamp: str = "window_end",
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

    candidates: list[Candidate] = []
    for on in threshold_on_grid:
        for off in threshold_off_grid:
            if off >= on:
                continue
            params = PostprocessParams(
                method=method, ema_alpha=ema_alpha, threshold_on=on, threshold_off=off,
                run_length=run_length, event_merge_gap_s=event_merge_gap_s,
                alarm_timestamp=alarm_timestamp, window_s=window_s,
            )
            total_events = total_matched = total_false_alarms = 0
            total_exposure = 0.0
            all_delays: list[float] = []
            for end_sec_by_edf, scores_by_edf, events_by_edf, exposure_hours_by_edf in val_folds:
                events, matched, false_alarms, exposure, delays = _score_candidate_totals(
                    end_sec_by_edf, scores_by_edf, events_by_edf, exposure_hours_by_edf, params
                )
                total_events += events
                total_matched += matched
                total_false_alarms += false_alarms
                total_exposure += exposure
                all_delays.extend(delays)

            sensitivity = total_matched / total_events if total_events else 0.0
            far = total_false_alarms / total_exposure if total_exposure else 0.0
            # A combination that detects nothing has no delay to speak of;
            # ranking it as infinitely fast would let `min_delay` pick it.
            mean_delay = float(np.mean(all_delays)) if all_delays else float("inf")
            candidates.append((params, sensitivity, far, mean_delay))

    if not candidates:
        raise RuntimeError(
            "threshold search produced no valid (on, off) combination "
            f"(grids: on={threshold_on_grid}, off={threshold_off_grid})"
        )

    params, sensitivity, far, mean_delay = _select_best(
        candidates, far_weight, far_cap_per_hour, objective, sensitivity_floor
    )
    return FrozenPostprocessParams(
        params=params,
        val_sensitivity=sensitivity,
        val_far_per_hour=far,
        val_delay_mean_s=mean_delay,
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
    objective: str = "max_sensitivity",
    sensitivity_floor: float | None = None,
    window_s: float = 0.0,
    alarm_timestamp: str = "window_end",
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
        objective=objective, sensitivity_floor=sensitivity_floor,
        window_s=window_s, alarm_timestamp=alarm_timestamp,
    )


def save_frozen_params(frozen: FrozenPostprocessParams, path: str) -> None:
    Path(path).write_text(json.dumps(frozen.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_frozen_params(path: str) -> FrozenPostprocessParams:
    return FrozenPostprocessParams.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
