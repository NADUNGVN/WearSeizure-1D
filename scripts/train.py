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
from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import load_folds, subject_from_fold_id
from wearseizure.models.factory import build_model
from wearseizure.training.distill import (
    get_or_train_fold_teacher_logits,
    pretrained_teacher_logits_fn,
)
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


log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)

    # DataLoader(num_workers>0) on Linux defaults to PyTorch's "file_descriptor"
    # strategy for passing tensors between worker processes, which consumes an
    # open file descriptor per shared tensor. Across ~66 folds x several
    # DataLoaders each, cleanup of the previous fold's workers can lag behind
    # creation of the next fold's (DataLoader iterators hold reference cycles
    # that rely on a full GC pass, not just refcounting), eventually exceeding
    # the OS's open-file limit ("OSError: Too many open files"). "file_system"
    # uses temp files instead and doesn't have this failure mode.
    #
    # Only with workers, and this is not a tidiness point. "file_system" spawns
    # torch_shm_manager, which must create a socket directory under the system
    # temp dir; on SERVER-04 it cannot, and every job there died with "could not
    # generate a random directory for manager socket". With num_workers=0 no
    # tensor is ever shared between processes, so the strategy is pure liability.
    # Set here rather than at import because it depends on cfg, and it still runs
    # before any DataLoader exists.
    if cfg.profile.get("num_workers", 0) > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")


    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    data_dir = cfg.data.generated_dir if cfg.data.name == "synthetic" else None
    raw_dir = cfg.data.raw_dir if cfg.data.name != "synthetic" else None
    records = load_records_from_manifest(manifest_df, data_dir=data_dir, raw_dir=raw_dir)

    # Version-lock the split to the manifest it was built from. PROTOCOL.md
    # calls splits version-locked by manifest hash; until now nothing checked
    # it, so a manifest rebuilt from changed data would have been paired with
    # stale folds silently -- and every comparison against earlier rows would
    # have been invalid without anything saying so.
    folds = load_folds(
        str(Path(cfg.split.folds_path)), expected_manifest_hash=hash_manifest(manifest_df)
    )
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


def _teacher_cfg(cfg, teacher_model_name: str):
    """`cfg` with its model group swapped for the lever-L8 teacher's.

    `build_model` reads the whole config, not a name, so the teacher's
    architecture has to be composed the same way Hydra composed the student's --
    from `configs/model/<name>.yaml` -- rather than guessed at.
    """
    teacher = OmegaConf.merge(cfg, {})
    model_yaml = Path(__file__).resolve().parents[1] / "configs" / "model" / f"{teacher_model_name}.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError(f"lever L8: no model config at {model_yaml}")
    teacher.model = OmegaConf.load(model_yaml)
    return teacher


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
    # The pre-training cache is normally tagged with the run, because a lever
    # that changes pre-training produces genuinely different initialisations.
    # But some levers do not touch it: L3 changes only the fine-tuning loss, so
    # tagging its cache would retrain 13 initialisations per model per seed to
    # arrive at bit-identical weights -- about forty hours to reproduce what is
    # already on disk. `share_cache_with_control` says so explicitly rather than
    # leaving it to whoever reads the run time and wonders.
    #
    # It must stay opt-in: turning it on for a lever that DOES change
    # pre-training would silently reuse the control's initialisations and the
    # experiment would measure nothing.
    pretrain_tag = (
        "" if pretrain_cfg.get("share_cache_with_control", False) else run_tag_from_cfg(cfg)
    )
    pretrain_dir = pretrain_cache_dir(
        cfg.profile.artifacts_dir, cfg.model.name, cfg.window.name, seed, pretrain_tag
    )
    # Lever L3. Teachers are cached by FOLD, not by seed: the teacher depends on
    # the fold's train partition and window geometry, none of which the student's
    # seed touches. Caching per seed would cost three times as much for three
    # answers to the same question, and would stop a seed-to-seed comparison
    # from isolating the student.
    distill_cfg = cfg.train.get("distill", {})
    use_distill = bool(distill_cfg.get("enabled", False))
    l8_teacher_model = distill_cfg.get("teacher_model") or ""
    teacher_run_tag = str(distill_cfg.get("teacher_run_tag") or "")
    teacher_seed = distill_cfg.get("teacher_seed")
    teacher_seed = seed if teacher_seed is None else int(teacher_seed)
    teacher_dir = (
        Path(cfg.profile.artifacts_dir) / "teacher" / cfg.window.name
        / (run_tag_from_cfg(cfg) or "base")
    )
    if use_distill and not l8_teacher_model:
        # Only the L3 teacher needs a multi-channel montage. L8's teacher is a
        # finished single-channel run, so it works on any dataset the student
        # works on -- and that is what makes it smoke-testable off the server.
        if cfg.data.name == "synthetic":
            raise ValueError(
                "train.distill.enabled=true needs real multi-channel EDFs; the synthetic "
                "generator produces one channel per record"
            )
        log.info(
            f"lever L3 ENABLED: teacher logits cached under {teacher_dir}; "
            f"alpha={distill_cfg.get('alpha')} temperature={distill_cfg.get('temperature')}"
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
                model_selection=cfg.train.get("model_selection", "val_loss"),
                extra_manifest_df=extra_manifest_df,
            )
            model.load_state_dict(init_state)
            fold_lr = finetune_lr
            pretrained_from = subject_id

        teacher_logits = None
        teacher_logits_fn = None
        if use_distill and l8_teacher_model:
            # Lever L8. The teacher is a finished run, not a newly trained
            # model: it has to be the checkpoint that produced the number being
            # distilled, because 0.9726 depends on cohort pre-training that a
            # freshly trained per-fold teacher would not have.
            teacher_logits_fn = pretrained_teacher_logits_fn(
                artifacts_dir=cfg.profile.artifacts_dir,
                teacher_model_name=l8_teacher_model,
                build_teacher_model=lambda: build_model(_teacher_cfg(cfg, l8_teacher_model)),
                split_name=cfg.split.name,
                window_name=cfg.window.name,
                seed=teacher_seed,
                run_tag=teacher_run_tag,
                fold_id=fold.fold_id,
                batch_size=cfg.train.batch_size,
                device=cfg.profile.device,
                num_workers=cfg.profile.get("num_workers", 0),
            )
            if teacher_logits_fn is None:
                # Falling back to no distillation would make this arm a mixture
                # of two experiments, with nothing in the metrics to say so.
                raise FileNotFoundError(
                    f"lever L8: no {l8_teacher_model} checkpoint for fold {fold.fold_id} "
                    f"(seed {teacher_seed}, tag {teacher_run_tag or '<none>'}). "
                    "Run that arm to completion before distilling from it."
                )
        elif use_distill:
            teacher_logits = get_or_train_fold_teacher_logits(
                records=records,
                manifest_df=manifest_df,
                raw_dir=cfg.data.raw_dir,
                fold=fold,
                cache_dir=teacher_dir,
                window_s=cfg.window.window_s,
                stride_s=cfg.window.stride_s,
                epochs=distill_cfg.get("teacher_epochs", 30),
                lr=distill_cfg.get("teacher_lr", 1e-3),
                weight_decay=cfg.train.weight_decay,
                batch_size=cfg.train.batch_size,
                device=cfg.profile.device,
                early_stopping_patience=distill_cfg.get("teacher_patience", 8),
                num_workers=cfg.profile.get("num_workers", 0),
                class_balanced_sampling=cfg.train.class_balanced_sampling,
                force=distill_cfg.get("force", False),
                single_channel=bool(distill_cfg.get("single_channel_teacher", False)),
            )

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
            model_selection=cfg.train.get("model_selection", "val_loss"),
            teacher_logits=teacher_logits,
            teacher_logits_fn=teacher_logits_fn,
            distill_alpha=float(distill_cfg.get("alpha", 0.0)) if use_distill else 0.0,
            distill_temperature=float(distill_cfg.get("temperature", 2.0)),
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
                    "distilled": bool(use_distill),
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
