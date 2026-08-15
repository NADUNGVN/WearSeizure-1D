"""Per-patient failure analysis for the worst-patient gate (memo 5.3:
"Worst-patient gate: ... Neu vi pham, phan tich failure thay vi che bang
average"). Reads the same per-fold *.metrics.json files evaluate.py
aggregates and, for every patient flagged below the worst-patient gates
(Table 6), prints a full per-fold breakdown -- including segment-level
AUROC, so a genuinely weak signal (low AUROC) can be told apart from a
small-sample-size threshold-selection artifact (good AUROC, but too few
validation events to pick a reliable threshold).

Read-only: does not train, retrain, or modify anything. Every patient stays
in personalized-mode evaluation regardless of what this flags -- the point
is diagnosis, not exclusion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
from omegaconf import DictConfig

from wearseizure.eval.report import load_gates
from wearseizure.utils.logging import get_logger

log = get_logger(__name__)


def _subject_from_fold_id(fold_id: str, strategy: str) -> str:
    prefix, _, rest = fold_id.partition("__")
    return rest if strategy == "zero_shot_loso_subject" else prefix


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.profile.artifacts_dir) / cfg.model.name / cfg.split.name / cfg.window.name
    fold_paths = sorted(run_dir.glob("*.metrics.json"))
    if not fold_paths:
        raise FileNotFoundError(f"no *.metrics.json in {run_dir} -- run train.py first")

    gates_path = Path(__file__).resolve().parent.parent / "configs" / "eval" / "gates.yaml"
    gates = load_gates(str(gates_path))
    worst_sens_min = gates["worst_patient_sensitivity"]["minimum"]
    worst_far_min = gates["worst_patient_far_per_hour"]["minimum"]

    by_patient: dict[str, list[dict]] = {}
    all_aurocs: list[float] = []
    for p in fold_paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        subject = _subject_from_fold_id(d["fold_id"], cfg.split.strategy)
        by_patient.setdefault(subject, []).append(d)
        all_aurocs.append(d["test_segment_metrics"]["auroc"])

    cohort_mean_auroc = sum(all_aurocs) / len(all_aurocs)
    log.info(f"cohort mean segment AUROC: {cohort_mean_auroc:.3f} across {len(fold_paths)} folds")

    flagged: list[str] = []
    for subject, folds in sorted(by_patient.items()):
        n_events = sum(f["test_event_metrics"]["n_events"] for f in folds)
        n_matched = sum(f["test_event_metrics"]["n_matched"] for f in folds)
        n_false_alarms = sum(f["test_event_metrics"]["n_false_alarms"] for f in folds)
        exposure = sum(f["test_event_metrics"]["exposure_hours"] for f in folds)
        sensitivity = n_matched / n_events if n_events else float("nan")
        far = n_false_alarms / exposure if exposure else float("nan")
        patient_auroc = sum(f["test_segment_metrics"]["auroc"] for f in folds) / len(folds)

        below_sens = not math.isnan(sensitivity) and sensitivity < worst_sens_min
        below_far = not math.isnan(far) and far > worst_far_min

        if below_sens or below_far:
            flagged.append(subject)
            reason = []
            if below_sens:
                reason.append(f"sensitivity {sensitivity:.2f} < gate minimum {worst_sens_min}")
            if below_far:
                reason.append(f"FAR {far:.2f}/h > gate minimum {worst_far_min}/h")
            log.info(
                f"=== FLAGGED: {subject} ({'; '.join(reason)}) -- n_events={n_events}, "
                f"n_folds={len(folds)}, mean segment AUROC={patient_auroc:.3f} "
                f"(cohort mean {cohort_mean_auroc:.3f}) ==="
            )
            for f in sorted(folds, key=lambda x: x["fold_id"]):
                tm, fp, seg = f["test_event_metrics"], f["frozen_postprocess"], f["test_segment_metrics"]
                log.info(
                    f"    fold {f['fold_id']}: n_events={tm['n_events']} n_matched={tm['n_matched']} "
                    f"delays={[round(x, 1) for x in tm['delays_s']]} | "
                    f"val_sensitivity={fp['val_sensitivity']:.2f} val_far={fp['val_far_per_hour']:.2f} "
                    f"threshold_on/off={fp['params']['threshold_on']}/{fp['params']['threshold_off']} | "
                    f"segment_auroc={seg['auroc']:.3f} segment_auprc={seg['auprc']:.3f}"
                )
        else:
            log.info(
                f"{subject}: sensitivity={sensitivity:.2f} far={far:.2f} n_events={n_events} "
                f"segment_auroc={patient_auroc:.3f} (within worst-patient gate)"
            )

    log.info(
        f"{len(flagged)}/{len(by_patient)} patients flagged below the worst-patient gate: {flagged}. "
        "Compare each patient's segment_auroc to the cohort mean above: close-to-cohort AUROC with a "
        "low n_events points at a small-sample threshold-selection artifact (memo 5.1 fits threshold "
        "on val only, and a 1-2 event validation set is inherently noisy); a patient-specific AUROC "
        "well below the cohort mean points at a genuinely harder signal for that patient/channel."
    )


if __name__ == "__main__":
    main()
