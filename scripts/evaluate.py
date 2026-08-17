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
from wearseizure.utils.profile_guard import check_profile_data_pairing

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

log = get_logger(__name__)


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


def _budget_from_frozen_params(fold_dicts: list[dict], cfg) -> "object":
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    run_dir = Path(cfg.profile.artifacts_dir) / cfg.model.name / cfg.split.name / cfg.window.name
    fold_dicts = _load_fold_metrics(run_dir)
    per_patient = _aggregate_per_patient(fold_dicts, cfg.split.strategy)

    budget = _budget_from_frozen_params(fold_dicts, cfg)

    report = build_report(per_patient, budget=budget)
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

    gates_path = Path(__file__).resolve().parent.parent / "configs" / "eval" / "gates.yaml"
    gates = load_gates(str(gates_path))
    gate_results = check_gates(flatten_for_gates(report), gates)
    for key, result in gate_results.items():
        log.info(f"gate {key}: value={result['value']:.4f} level={result['level']}")

    if cfg.profile.enforce_gates:
        failing = {k: v for k, v in gate_results.items() if v["level"] == "below_minimum"}
        if failing:
            raise RuntimeError(f"Gate check failed (below minimum): {failing}")
    else:
        log.info("enforce_gates=false (synthetic/local profile): gate failures are reported, not fatal")


if __name__ == "__main__":
    main()
