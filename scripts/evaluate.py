"""Aggregate per-fold metrics (written by train.py) into per-patient event
metrics, build the macro/micro/CI report, and check it against Table 6 gates.
Gate failures only raise when `profile.enforce_gates` is true (server profile
with real data) -- on synthetic data they are reported but never fatal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from wearseizure.data.splits import subject_from_fold_id
from wearseizure.eval.delay_budget import delay_budget
from wearseizure.eval.metrics_event import EventMetrics
from wearseizure.eval.report import (
    build_report,
    check_gates,
    flatten_for_gates,
    load_gates,
    save_report,
)
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import (
    fold_run_dir,
    run_tag_from_cfg,
    seeds_from_cfg,
    warn_if_legacy_artifacts,
)
from wearseizure.utils.profile_guard import check_profile_data_pairing

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

log = get_logger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_fold_metrics(run_dir: Path) -> list[dict]:
    fold_metrics = sorted(run_dir.glob("*.metrics.json"))
    if not fold_metrics:
        raise FileNotFoundError(f"no *.metrics.json found in {run_dir} -- run train.py first")
    return [json.loads(p.read_text(encoding="utf-8")) for p in fold_metrics]


def _aggregate_per_patient(fold_dicts: list[dict], strategy: str) -> dict[str, EventMetrics]:
    by_patient: dict[str, list[dict]] = {}
    for d in fold_dicts:
        subject = subject_from_fold_id(d["fold_id"], strategy)
        by_patient.setdefault(subject, []).append(d["test_event_metrics"])

    per_patient: dict[str, EventMetrics] = {}
    for subject, metrics_list in by_patient.items():
        n_events = sum(m["n_events"] for m in metrics_list)
        n_matched = sum(m["n_matched"] for m in metrics_list)
        n_false_alarms = sum(m["n_false_alarms"] for m in metrics_list)
        exposure_hours = sum(m["exposure_hours"] for m in metrics_list)
        delays = [d for m in metrics_list for d in m["delays_s"]]
        per_patient[subject] = EventMetrics(
            n_events=n_events,
            n_matched=n_matched,
            n_missed=n_events - n_matched,
            n_false_alarms=n_false_alarms,
            sensitivity=(n_matched / n_events if n_events else float("nan")),
            far_per_hour=(n_false_alarms / exposure_hours if exposure_hours else float("nan")),
            delays_s=delays,
            exposure_hours=exposure_hours,
        )
    return per_patient


def _budget_from_frozen_params(fold_dicts: list[dict], cfg) -> object:
    """Delay floor from the postprocess params the folds ACTUALLY used.

    Taking these from `cfg.postprocess` instead is a trap: `rethreshold.py`
    accepts `postprocess.run_length=1 postprocess.ema_alpha=0.5` overrides and
    writes the result into the same `*.metrics.json` files, but `evaluate.py`
    is then usually run without repeating those overrides. It would compute the
    floor from the config defaults and silently report a floor -- and therefore
    a "model reaction" -- for a configuration that never ran. That happened on
    the first (run_length=1, ema_alpha=0.5) sweep: a 5.0s floor was reported as
    13.0s, overstating model reaction by 8 seconds.

    `run_length`, `ema_alpha` and `alarm_timestamp` come from the frozen params
    because those produced the alarms. `window_s`/`stride_s` come from the
    window config, which is what the windows were actually cut with.
    """
    frozen = [d["frozen_postprocess"]["params"] for d in fold_dicts if "frozen_postprocess" in d]
    if not frozen:
        log.warning("no frozen_postprocess in fold metrics -- falling back to cfg.postprocess for the delay floor")
        params = {
            "run_length": cfg.postprocess.get("run_length", 1),
            "ema_alpha": cfg.postprocess.get("ema_alpha", 0.0),
            "alarm_timestamp": cfg.postprocess.get("alarm_timestamp", "window_end"),
        }
    else:
        distinct = {(p.get("run_length", 1), p.get("ema_alpha", 0.0), p.get("alarm_timestamp", "window_end"))
                    for p in frozen}
        if len(distinct) > 1:
            raise RuntimeError(
                "folds were thresholded under different postprocess settings, so a single delay "
                f"floor is meaningless: {sorted(distinct)}. Re-run rethreshold.py over all folds."
            )
        params = frozen[0]

    for key, cfg_value in (("run_length", cfg.postprocess.get("run_length", 1)),
                           ("ema_alpha", cfg.postprocess.get("ema_alpha", 0.0))):
        used = params.get(key)
        if used is not None and used != cfg_value:
            log.warning(
                f"postprocess.{key}: folds were thresholded with {used}, current config says "
                f"{cfg_value}. Using {used} (what actually ran) for the delay floor."
            )

    return delay_budget(
        window_s=cfg.window.window_s,
        stride_s=cfg.window.stride_s,
        run_length=params.get("run_length", 1),
        ema_alpha=params.get("ema_alpha", 0.0),
        alarm_timestamp=params.get("alarm_timestamp", "window_end"),
    )


def _resolve_gates_path(cfg) -> Path:
    """Which gate table to score against.

    Was hardcoded to `configs/eval/gates.yaml`. Made overridable so the
    proposed v2 table (`configs/eval/gates_v2_proposed.yaml`) can be scored
    without editing code -- the two tables disagree about what this project is
    trying to prove, so which one a number was checked against has to be
    visible in the run's overrides. Default is unchanged.
    """
    configured = cfg.eval.get("gates_path")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else _CONFIGS_DIR.parent / path
    return _CONFIGS_DIR / "eval" / "gates.yaml"


def _evaluate_one_seed(cfg, seed: int, gates: dict, min_events_to_gate: int | None) -> tuple[dict, dict]:
    """Build and score the report for one seed. Returns (report, gate_results)."""
    run_dir = fold_run_dir(
        cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name, cfg.window.name, seed, run_tag_from_cfg(cfg)
    )
    warn_if_legacy_artifacts(run_dir, log)
    fold_dicts = _load_fold_metrics(run_dir)
    per_patient = _aggregate_per_patient(fold_dicts, cfg.split.strategy)

    budget = _budget_from_frozen_params(fold_dicts, cfg)

    report = build_report(per_patient, budget=budget, min_events_to_gate=min_events_to_gate)
    report_path = run_dir / "report.json"
    save_report(report, str(report_path))
    log.info(f"report written to {report_path}")
    log.info(
        f"macro sensitivity={report['macro']['sensitivity_macro']:.3f} "
        f"FAR/h={report['macro']['far_per_hour_macro']:.3f} "
        f"exposure={report['micro']['exposure_hours']:.1f}h"
    )
    # Delay is meaningless without its floor: with the shipped defaults
    # (w4s_stride1s, run_length=3, ema_alpha=0.125) the floor alone is 13.0s,
    # against a v1 gate of 5.0s. See docs/RESEARCH_REALITY_CHECK.md section 3.
    log.info(
        f"delay mean={report['delay']['mean_s']:.2f}s "
        f"= floor {budget.floor_s:.2f}s "
        f"(window {budget.window_term_s:.2f} + run-length {budget.run_length_term_s:.2f} "
        f"+ EMA {budget.ema_term_s:.2f}) "
        f"+ model reaction {report['delay']['model_reaction_mean_s']:.2f}s"
    )
    if budget.floor_s > 0:
        log.info(
            f"delay under window_start convention (for comparison with published "
            f"single-channel baselines): mean="
            f"{report['delay']['window_start_convention']['mean_s']:.2f}s"
        )

    gate_results = check_gates(flatten_for_gates(report), gates)
    for key, result in gate_results.items():
        log.info(f"gate {key}: value={result['value']:.4f} level={result['level']}")
    worst = report["worst_patient"]
    if worst.get("patients_below_event_floor"):
        exempt = worst["patients_below_event_floor"]
        if worst["sensitivity_gated_patient"] is None:
            log.warning(
                f"worst-patient sensitivity NOT gated: no patient has >= "
                f"{worst['min_events_to_gate']} seizures (all of {exempt} are below the floor)"
            )
        else:
            log.info(
                f"worst-patient sensitivity gated on {worst['sensitivity_gated_patient']} "
                f"({worst['sensitivity_gated_patient_n_events']} events); "
                f"exempt (too few seizures): {exempt}"
            )
        log.info("small-sample patients are in report['small_sample_patients'] with exact binomial intervals")

    return report, gate_results


def _multiseed_summary(reports_by_seed: dict[int, dict]) -> dict:
    """Mean and sample standard deviation of every gated metric across seeds.

    This is the whole point of lever L7. The three best configurations to date
    sit at 0.9218 / 0.9256 / 0.9359 macro sensitivity -- 1.4pp apart on 77
    seizures, i.e. about one seizure -- so without a spread across seeds there
    is no basis for preferring any of them, and no way to state the paper's
    central claim (equivalence-or-better vs the reproduced baselines) as
    anything but a point estimate.

    `ddof=1`: these seeds are a sample of the seed distribution, not the whole
    of it. With a single seed the std is undefined and reported as NaN rather
    than 0, which would read as "perfectly reproducible".
    """
    keys = sorted({k for r in reports_by_seed.values() for k in flatten_for_gates(r)})
    summary = {}
    for key in keys:
        values = [flatten_for_gates(r).get(key) for r in reports_by_seed.values()]
        values = [float(v) for v in values if v is not None and not np.isnan(float(v))]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": int(arr.size),
            "per_seed": {str(s): float(flatten_for_gates(r).get(key, float("nan")))
                         for s, r in reports_by_seed.items()},
        }
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)

    # Gates are loaded BEFORE any report is built: the worst-patient rule needs
    # `min_events_to_gate` from the gates file to know which patients a
    # sensitivity threshold is allowed to land on (see
    # eval/metrics_event.worst_patient). v1 gates declare no floor, so this
    # resolves to None and the report is byte-identical to before.
    gates_path = _resolve_gates_path(cfg)
    gates = load_gates(str(gates_path))
    min_events_to_gate = (gates.get("worst_patient_sensitivity") or {}).get("min_events_to_gate")
    log.info(
        f"gates: {gates_path}"
        + (f" (worst-patient floor: >={min_events_to_gate} events)" if min_events_to_gate else "")
    )

    seeds = seeds_from_cfg(cfg)
    reports: dict[int, dict] = {}
    gate_results_by_seed: dict[int, dict] = {}
    for seed in seeds:
        if len(seeds) > 1:
            log.info(f"--- seed {seed} ---")
        reports[seed], gate_results_by_seed[seed] = _evaluate_one_seed(
            cfg, seed, gates, min_events_to_gate
        )

    if len(seeds) > 1:
        summary = _multiseed_summary(reports)
        # The tag has to be here too. Without it the per-seed reports land in the
        # tagged directory while the summary lands in the UNTAGGED one -- so a
        # lever-L5 run silently overwrote the control arm's summary with its own
        # numbers, and the only symptom was a comparison table where the control
        # had quietly changed.
        summary_path = (
            fold_run_dir(cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name,
                         cfg.window.name, seeds[0], run_tag_from_cfg(cfg)).parent
            / "report_multiseed.json"
        )
        save_report({"seeds": seeds, "metrics": summary}, str(summary_path))
        log.info(f"multi-seed summary written to {summary_path}")
        for key, stats in summary.items():
            log.info(
                f"[{len(seeds)} seeds] {key}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
                f"(min {stats['min']:.4f}, max {stats['max']:.4f})"
            )
        # Gate the MEAN, not any individual seed: picking the best seed is the
        # same selection-on-the-evaluation-set mistake docs/PROTOCOL.md forbids
        # for thresholds.
        gate_results = check_gates({k: v["mean"] for k, v in summary.items()}, gates)
        for key, result in gate_results.items():
            log.info(f"gate (mean of {len(seeds)} seeds) {key}: value={result['value']:.4f} level={result['level']}")
    else:
        gate_results = gate_results_by_seed[seeds[0]]

    if cfg.profile.enforce_gates:
        failing = {k: v for k, v in gate_results.items() if v["level"] == "below_minimum"}
        if failing:
            raise RuntimeError(f"Gate check failed (below minimum): {failing}")
    else:
        log.info("enforce_gates=false (synthetic/local profile): gate failures are reported, not fatal")


if __name__ == "__main__":
    main()
