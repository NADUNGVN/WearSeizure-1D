"""Re-run threshold selection using validation evidence POOLED across every
fold belonging to the same patient, instead of each fold picking its own
threshold from just its own (often tiny) validation set.

Motivation (memo 5.1/5.3, and the failure_analysis.py findings): patients
with very few total seizures (e.g. 3) give each of their personalized folds
a validation set with just 1 event -- too noisy to reliably pick
threshold_on/threshold_off from alone. Pooling that patient's several folds'
validation evidence together (each fold's val EDFs still scored by *that
fold's own* trained model -- nothing here mixes model weights across folds)
gives the search more data points, without ever touching test data or
retraining.

Reads already-trained checkpoints from scripts/train.py (no training here).
Writes the same per-fold *.metrics.json format as train.py/rethreshold.py,
so scripts/evaluate.py and scripts/failure_analysis.py work unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from wearseizure.data.dataset import build_fold_datasets
from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import load_manifest
from wearseizure.data.splits import load_folds, subject_from_fold_id
from wearseizure.eval.metrics_event import EventMetrics, compute_event_metrics
from wearseizure.eval.metrics_segment import compute_segment_metrics
from wearseizure.models.factory import build_model
from wearseizure.postprocess.pipeline import run_postprocess
from wearseizure.training.engine_baseline import group_by_edf, score_partition
from wearseizure.training.threshold_selection import fit_threshold_on_val_pooled
from wearseizure.utils.logging import get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    data_dir = cfg.data.generated_dir if cfg.data.name == "synthetic" else None
    raw_dir = cfg.data.raw_dir if cfg.data.name != "synthetic" else None
    records = load_records_from_manifest(manifest_df, data_dir=data_dir, raw_dir=raw_dir)

    folds = load_folds(str(Path(cfg.split.folds_path)))
    run_dir = Path(cfg.profile.artifacts_dir) / cfg.model.name / cfg.split.name / cfg.window.name
    threshold_grid = cfg.postprocess.get("threshold_search", OmegaConf.create({}))
    threshold_on_grid = list(threshold_grid.get("on_grid", [cfg.postprocess.get("threshold", 0.5)]))
    threshold_off_grid = list(threshold_grid.get("off_grid", [cfg.postprocess.get("threshold", 0.5) - 0.1]))

    # Pass 1: score every fold's val + test partitions with that fold's own
    # checkpoint. Nothing here picks a threshold yet.
    per_fold_data = {}
    for fold in folds:
        checkpoint_path = run_dir / f"{fold.fold_id}.pt"
        if not checkpoint_path.exists():
            log.warning(f"no checkpoint for {fold.fold_id} at {checkpoint_path}, skipping (run train.py first)")
            continue

        model = build_model(cfg)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model.to(cfg.profile.device)

        datasets, _band, _normalizer = build_fold_datasets(records, fold, cfg.window.window_s, cfg.window.stride_s)
        val_ds, test_ds = datasets["val"], datasets["test"]

        val_scores, _val_labels, val_end_sec, val_edf_ids = score_partition(
            model, val_ds, cfg.profile.device, batch_size=cfg.train.batch_size,
            num_workers=cfg.profile.get("num_workers", 0),
        )
        val_group = group_by_edf(val_end_sec, val_scores, val_edf_ids, records, fold.val_edf_ids)

        test_scores, test_labels, test_end_sec, test_edf_ids = score_partition(
            model, test_ds, cfg.profile.device, batch_size=cfg.train.batch_size,
            num_workers=cfg.profile.get("num_workers", 0),
        )
        test_group = group_by_edf(test_end_sec, test_scores, test_edf_ids, records, fold.test_edf_ids)

        per_fold_data[fold.fold_id] = {
            "fold": fold, "val_group": val_group, "test_scores": test_scores,
            "test_labels": test_labels, "test_group": test_group,
        }
        log.info(f"scored fold {fold.fold_id} (val + test, no threshold yet)")

    # Pass 2: pool val evidence per patient, pick ONE threshold per patient.
    by_patient: dict[str, list[str]] = {}
    for fold_id in per_fold_data:
        subject = subject_from_fold_id(fold_id, cfg.split.strategy)
        by_patient.setdefault(subject, []).append(fold_id)

    frozen_by_patient = {}
    for subject, fold_ids in by_patient.items():
        val_folds = [per_fold_data[fid]["val_group"] for fid in fold_ids]
        frozen = fit_threshold_on_val_pooled(
            val_folds=val_folds,
            method=cfg.postprocess.method, ema_alpha=cfg.postprocess.get("ema_alpha", 0.125),
            run_length=cfg.postprocess.get("run_length", 1),
            event_merge_gap_s=cfg.postprocess.get("event_merge_gap_s", 0.0),
            threshold_on_grid=threshold_on_grid, threshold_off_grid=threshold_off_grid,
            group_id=subject, far_cap_per_hour=cfg.postprocess.get("far_cap_per_hour"),
        )
        frozen_by_patient[subject] = frozen
        log.info(
            f"patient {subject} ({len(fold_ids)} folds pooled): threshold_on/off="
            f"{frozen.params.threshold_on}/{frozen.params.threshold_off} "
            f"val_sensitivity={frozen.val_sensitivity:.2f} val_far={frozen.val_far_per_hour:.2f}"
        )

    # Pass 3: apply each patient's shared threshold to each of their folds'
    # own cached test scores, write metrics.json in the standard format.
    for fold_id, data in per_fold_data.items():
        fold = data["fold"]
        subject = subject_from_fold_id(fold_id, cfg.split.strategy)
        frozen = frozen_by_patient[subject]
        test_end_sec_by_edf, test_scores_by_edf, test_events_by_edf, test_exposure_by_edf = data["test_group"]

        all_events, delays = [], []
        total_matched = total_false_alarms = 0
        total_exposure = 0.0
        for edf_id in fold.test_edf_ids:
            alarms = run_postprocess(test_end_sec_by_edf[edf_id], test_scores_by_edf[edf_id], frozen.params)
            edf_events = test_events_by_edf[edf_id]
            m = compute_event_metrics(edf_events, alarms, test_exposure_by_edf[edf_id])
            total_matched += m.n_matched
            total_false_alarms += m.n_false_alarms
            total_exposure += m.exposure_hours
            delays.extend(m.delays_s)
            all_events.extend(edf_events)

        n_events = len(all_events)
        test_event_metrics = EventMetrics(
            n_events=n_events, n_matched=total_matched, n_missed=n_events - total_matched,
            n_false_alarms=total_false_alarms,
            sensitivity=(total_matched / n_events if n_events else float("nan")),
            far_per_hour=(total_false_alarms / total_exposure if total_exposure else float("nan")),
            delays_s=delays, exposure_hours=total_exposure,
        )
        test_segment_metrics = compute_segment_metrics(data["test_labels"], data["test_scores"])

        metrics_path = run_dir / f"{fold_id}.metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "fold_id": fold_id,
                    "held_out_key": fold.held_out_key,
                    "frozen_postprocess": frozen.to_dict(),
                    "test_event_metrics": asdict(test_event_metrics),
                    "test_segment_metrics": asdict(test_segment_metrics),
                },
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
        log.info(
            f"fold {fold_id} (pooled threshold from patient {subject}): "
            f"test sensitivity={test_event_metrics.sensitivity:.3f} "
            f"FAR/h={test_event_metrics.far_per_hour:.3f} -> {metrics_path}"
        )

    log.info(
        f"re-thresholded {len(per_fold_data)} folds across {len(by_patient)} patients (pooled) for "
        f"model={cfg.model.name}, split={cfg.split.name} -> {run_dir}"
    )


if __name__ == "__main__":
    main()
