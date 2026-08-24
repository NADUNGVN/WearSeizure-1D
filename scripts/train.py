"""Train across every fold of the configured split strategy, freezing
postprocess thresholds on validation and evaluating on each fold's continuous
test partition. Per-fold metrics are saved so `evaluate.py` can aggregate
per-patient without retraining.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import load_manifest
from wearseizure.data.splits import load_folds, subject_from_fold_id
from wearseizure.models.factory import build_model
from wearseizure.training.engine_baseline import run_fold
from wearseizure.training.pretrain import get_or_train_cohort_init, load_pretrain_corpus
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import (
    ensure_dir,
    fold_run_dir,
    pretrain_cache_dir,
    run_tag_from_cfg,
    seeds_from_cfg,
    warn_if_legacy_artifacts,
)
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.seeding import seed_everything

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

# DataLoader(num_workers>0) on Linux defaults to PyTorch's "file_descriptor"
# strategy for passing tensors between worker processes, which consumes an
# open file descriptor per shared tensor. Across ~66 folds x several
# DataLoaders each, cleanup of the previous fold's workers can lag behind
# creation of the next fold's (DataLoader iterators hold reference cycles
# that rely on a full GC pass, not just refcounting), eventually exceeding
# the OS's open-file limit ("OSError: Too many open files"). "file_system"
# uses temp files instead and doesn't have this failure mode.
torch.multiprocessing.set_sharing_strategy("file_system")

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)

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

    threshold_grid = cfg.postprocess.get("threshold_search", OmegaConf.create({}))
    force_retrain = cfg.train.get("force_retrain", False)

    seeds = seeds_from_cfg(cfg)
    if len(seeds) > 1:
        log.info(
            f"multi-seed run (lever L7): seeds={seeds}. Every fold is trained once per seed into "
            f"its own seed<N> directory; scripts/evaluate.py then reports mean +/- std across "
            f"them. Without this, differences of ~1 seizure out of 77 cannot be told from noise."
        )
    for seed in seeds:
        _run_one_seed(cfg, seed, records, manifest_df, folds, threshold_grid, force_retrain)


def _run_one_seed(cfg, seed, records, manifest_df, folds, threshold_grid, force_retrain) -> None:
    seed_everything(seed)

    # window.name is part of the path (not just model/split) because window_s/
    # stride_s changes what the training data itself looks like (memo 7.2
    # ablates window/stride) -- without this, switching window configs would
    # find the previous window's metrics.json already sitting there and
    # silently skip training instead of producing a real, comparable result.
    # seed<N> is there for the same reason: two seeds are two different runs.
    run_dir = fold_run_dir(
        cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name, cfg.window.name, seed, run_tag_from_cfg(cfg)
    )
    warn_if_legacy_artifacts(run_dir, log)
    ensure_dir(run_dir)

    # Lever L1 (docs/RESEARCH_REALITY_CHECK.md section 14): initialise each
    # fold from a model pre-trained on every OTHER patient, instead of from
    # random weights on one patient's handful of EDFs. Opt-in, so every result
    # recorded in docs/EXPERIMENT_LOG_G1a.md stays reproducible by default.
    pretrain_cfg = cfg.train.get("pretrain", {})
    use_pretrain = bool(pretrain_cfg.get("enabled", False))
    if use_pretrain and cfg.split.strategy != "patient_specific_loso_edf":
        log.warning(
            f"train.pretrain.enabled=true has no effect for split.strategy="
            f"{cfg.split.strategy!r}: that split already trains on other subjects. Ignoring."
        )
        use_pretrain = False
    pretrain_dir = pretrain_cache_dir(
        cfg.profile.artifacts_dir, cfg.model.name, cfg.window.name, seed, run_tag_from_cfg(cfg)
    )
    finetune_lr = cfg.train.get("finetune_lr", cfg.train.lr)
    extra_manifest_df, extra_records = (
        load_pretrain_corpus(cfg, log) if use_pretrain else (None, {})
    )
    if extra_records:
        records = {**records, **extra_records}
    if use_pretrain:
        log.info(
            f"cohort pre-training ENABLED: per-subject inits cached under {pretrain_dir}; "
            f"fine-tuning at lr={finetune_lr} (vs from-scratch lr={cfg.train.lr})"
        )

    n_trained = n_skipped = 0

    for fold in folds:
        metrics_path = run_dir / f"{fold.fold_id}.metrics.json"
        if metrics_path.exists() and not force_retrain:
            log.info(f"fold {fold.fold_id}: {metrics_path} already exists, skipping (train.force_retrain=true to redo)")
            n_skipped += 1
            continue

        log.info(f"training fold {fold.fold_id}")
        model = build_model(cfg)
        fold_lr = cfg.train.lr
        pretrained_from = None
        if use_pretrain:
            subject_id = subject_from_fold_id(fold.fold_id, cfg.split.strategy)
            init_state = get_or_train_cohort_init(
                records=records,
                manifest_df=manifest_df,
                held_out_subject=subject_id,
                model_factory=lambda: build_model(cfg),
                cache_dir=pretrain_dir,
                window_s=cfg.window.window_s,
                stride_s=cfg.window.stride_s,
                seed=seed,
                epochs=pretrain_cfg.get("epochs", cfg.train.epochs),
                lr=pretrain_cfg.get("lr", cfg.train.lr),
                weight_decay=cfg.train.weight_decay,
                batch_size=cfg.train.batch_size,
                device=cfg.profile.device,
                early_stopping_patience=pretrain_cfg.get(
                    "early_stopping_patience", cfg.train.early_stopping_patience
                ),
                num_workers=cfg.profile.get("num_workers", 0),
                val_subject_fraction=pretrain_cfg.get("val_subject_fraction", 0.2),
                class_balanced_sampling=cfg.train.class_balanced_sampling,
                force=pretrain_cfg.get("force", False),
                extra_manifest_df=extra_manifest_df,
            )
            model.load_state_dict(init_state)
            fold_lr = finetune_lr
            pretrained_from = subject_id

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
            lr=fold_lr,
            weight_decay=cfg.train.weight_decay,
            batch_size=cfg.train.batch_size,
            device=cfg.profile.device,
            class_balanced_sampling=cfg.train.class_balanced_sampling,
            early_stopping_patience=cfg.train.early_stopping_patience,
            num_workers=cfg.profile.get("num_workers", 0),
            far_cap_per_hour=cfg.postprocess.get("far_cap_per_hour"),
            objective=cfg.postprocess.get("objective", "max_sensitivity"),
            sensitivity_floor=cfg.postprocess.get("sensitivity_floor"),
            postprocess_alarm_timestamp=cfg.postprocess.get("alarm_timestamp", "window_end"),
            compile_mode=cfg.train.get("compile_mode"),
        )

        torch.save(result.model.state_dict(), run_dir / f"{fold.fold_id}.pt")
        metrics_path.write_text(
            json.dumps(
                {
                    "fold_id": fold.fold_id,
                    "held_out_key": fold.held_out_key,
                    # Provenance: a fold trained from a cohort init is not
                    # comparable to one trained from scratch, so the metrics
                    # file has to say which it was.
                    "pretrained_from_cohort_excluding": pretrained_from,
                    "lr": fold_lr,
                    "seed": seed,
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
        n_trained += 1

    log.info(
        f"seed {seed}: trained {n_trained} folds (skipped {n_skipped} already done) for "
        f"model={cfg.model.name}, split={cfg.split.name} -> {run_dir}"
    )


if __name__ == "__main__":
    main()
