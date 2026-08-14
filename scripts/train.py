"""Train across every fold of the configured split strategy, freezing
postprocess thresholds on validation and evaluating on each fold's continuous
test partition. Per-fold metrics are saved so `evaluate.py` can aggregate
per-patient without retraining.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import load_manifest
from wearseizure.data.splits import load_folds
from wearseizure.models.baselines import Compact1DBaseline, FrontiersBaseline2D
from wearseizure.models.wearseizure1d import WearSeizure1D
from wearseizure.training.engine_baseline import run_fold
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir
from wearseizure.utils.seeding import seed_everything

log = get_logger(__name__)

MODEL_FACTORIES = {
    "wearseizure1d": lambda cfg: WearSeizure1D(
        in_channels=cfg.model.in_channels,
        input_len=cfg.model.input_len,
        stem_out_channels=cfg.model.stem_out_channels,
        stage_out_channels=tuple(cfg.model.stage_out_channels),
        context_channels=cfg.model.context_channels,
        dilations=tuple(cfg.model.dilations),
        num_classes=cfg.model.num_classes,
    ),
    "baseline_frontiers2d": lambda cfg: FrontiersBaseline2D(
        in_channels=cfg.model.in_channels,
        input_len=cfg.model.input_len,
        branch_kernels=tuple(tuple(k) for k in cfg.model.branch_kernels),
        num_classes=cfg.model.num_classes,
    ),
    "baseline_compact1d_7k": lambda cfg: Compact1DBaseline(
        in_channels=cfg.model.in_channels, input_len=cfg.model.input_len, num_classes=cfg.model.num_classes
    ),
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    data_dir = cfg.data.generated_dir if cfg.data.name == "synthetic" else None
    raw_dir = cfg.data.raw_dir if cfg.data.name != "synthetic" else None
    records = load_records_from_manifest(manifest_df, data_dir=data_dir, raw_dir=raw_dir)

    folds = load_folds(str(Path(cfg.split.folds_path)))
    max_folds = cfg.train.get("max_folds")
    if max_folds is not None:
        log.warning(
            f"train.max_folds={max_folds}: running a SMOKE TEST on the first {max_folds} of "
            f"{len(folds)} folds, not a full run -- do not treat the resulting metrics as "
            "representative of the whole cohort."
        )
        folds = folds[:max_folds]

    model_factory = MODEL_FACTORIES.get(cfg.model.name)
    if model_factory is None:
        raise ValueError(f"unknown model {cfg.model.name!r}, expected one of {list(MODEL_FACTORIES)}")

    run_dir = ensure_dir(Path(cfg.profile.artifacts_dir) / cfg.model.name / cfg.split.name)
    threshold_grid = cfg.postprocess.get("threshold_search", OmegaConf.create({}))

    for fold in folds:
        log.info(f"training fold {fold.fold_id}")
        model = model_factory(cfg)
        result = run_fold(
            model=model,
            records=records,
            fold=fold,
            window_s=cfg.window.window_s,
            stride_s=cfg.window.stride_s,
            postprocess_method=cfg.postprocess.method,
            postprocess_ema_alpha=cfg.postprocess.get("ema_alpha", 0.125),
            postprocess_run_length=cfg.postprocess.get("run_length", 1),
            postprocess_event_merge_gap_s=cfg.postprocess.get("event_merge_gap_s", 0.0),
            threshold_on_grid=list(threshold_grid.get("on_grid", [cfg.postprocess.get("threshold", 0.5)])),
            threshold_off_grid=list(threshold_grid.get("off_grid", [cfg.postprocess.get("threshold", 0.5) - 0.1])),
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            batch_size=cfg.train.batch_size,
            device=cfg.profile.device,
            class_balanced_sampling=cfg.train.class_balanced_sampling,
            early_stopping_patience=cfg.train.early_stopping_patience,
        )

        torch.save(result.model.state_dict(), run_dir / f"{fold.fold_id}.pt")
        metrics_path = run_dir / f"{fold.fold_id}.metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "fold_id": fold.fold_id,
                    "held_out_key": fold.held_out_key,
                    "frozen_postprocess": result.frozen_postprocess.to_dict(),
                    "test_event_metrics": asdict(result.test_event_metrics),
                    "test_segment_metrics": asdict(result.test_segment_metrics),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        log.info(
            f"fold {fold.fold_id}: test sensitivity={result.test_event_metrics.sensitivity:.3f} "
            f"FAR/h={result.test_event_metrics.far_per_hour:.3f} -> {metrics_path}"
        )

    log.info(f"trained {len(folds)} folds for model={cfg.model.name}, split={cfg.split.name} -> {run_dir}")


if __name__ == "__main__":
    main()
