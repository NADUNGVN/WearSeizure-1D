"""Re-run threshold selection + evaluation for every fold using its already
trained checkpoint (from scripts/train.py), without re-training.

Use this after changing postprocess/threshold-search settings
(configs/postprocess/*.yaml) to see the effect cheaply -- training is by far
the expensive part of a full run; re-scoring + re-thresholding from a saved
checkpoint is comparatively fast. Overwrites the same
`<fold_id>.metrics.json` files scripts/train.py wrote, so scripts/evaluate.py
picks up the new results with no other changes.
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
from wearseizure.data.splits import load_folds
from wearseizure.models.factory import build_model
from wearseizure.training.engine_baseline import evaluate_fold
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import fold_run_dir
from wearseizure.utils.profile_guard import check_profile_data_pairing

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

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
    seed = int(cfg.seed)
    run_dir = fold_run_dir(
        cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name, cfg.window.name, seed
    )
    threshold_grid = cfg.postprocess.get("threshold_search", OmegaConf.create({}))

    n_done = n_missing = 0
    for fold in folds:
        checkpoint_path = run_dir / f"{fold.fold_id}.pt"
        if not checkpoint_path.exists():
            log.warning(f"no checkpoint for {fold.fold_id} at {checkpoint_path}, skipping (run train.py first)")
            n_missing += 1
            continue

        model = build_model(cfg)
        # weights_only=True: our checkpoints are only ever a state_dict (plain
        # tensors), so this is safe and avoids torch's FutureWarning about
        # unpickling arbitrary objects.
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

        result = evaluate_fold(
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
            batch_size=cfg.train.batch_size,
            device=cfg.profile.device,
            num_workers=cfg.profile.get("num_workers", 0),
            far_cap_per_hour=cfg.postprocess.get("far_cap_per_hour"),
            objective=cfg.postprocess.get("objective", "max_sensitivity"),
            sensitivity_floor=cfg.postprocess.get("sensitivity_floor"),
            postprocess_alarm_timestamp=cfg.postprocess.get("alarm_timestamp", "window_end"),
        )

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
        n_done += 1

    log.info(
        f"re-thresholded {n_done} folds ({n_missing} missing checkpoints) for "
        f"model={cfg.model.name}, split={cfg.split.name} -> {run_dir}"
    )


if __name__ == "__main__":
    main()
