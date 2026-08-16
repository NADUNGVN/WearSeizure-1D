"""Aggregate per-fold metrics (written by train.py) into per-patient event
metrics, build the macro/micro/CI report, and check it against Table 6 gates.
Gate failures only raise when `profile.enforce_gates` is true (server profile
with real data) -- on synthetic data they are reported but never fatal.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from wearseizure.data.splits import subject_from_fold_id
from wearseizure.eval.metrics_event import EventMetrics
from wearseizure.eval.report import (
    build_report,
    check_gates,
    flatten_for_gates,
    load_gates,
    save_report,
)
from wearseizure.utils.logging import get_logger

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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.profile.artifacts_dir) / cfg.model.name / cfg.split.name / cfg.window.name
    fold_dicts = _load_fold_metrics(run_dir)
    per_patient = _aggregate_per_patient(fold_dicts, cfg.split.strategy)

    report = build_report(per_patient)
    report_path = run_dir / "report.json"
    save_report(report, str(report_path))
    log.info(f"report written to {report_path}")
    log.info(
        f"macro sensitivity={report['macro']['sensitivity_macro']:.3f} "
        f"FAR/h={report['macro']['far_per_hour_macro']:.3f} "
        f"exposure={report['micro']['exposure_hours']:.1f}h"
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
