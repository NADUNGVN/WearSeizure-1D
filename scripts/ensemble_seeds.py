"""Score every fold with an inference-time ensemble of already trained seeds.

Each fold's members are that fold's checkpoints from `train.seeds`, under
`train.run_tag` -- for the headline configuration, the three L8 seeds. Their
logits are averaged (`wearseizure.models.ensemble.LogitMeanEnsemble`) and the
ensemble then goes through exactly the path `rethreshold.py` uses for a single
model: thresholds searched on the fold's VALIDATION partition, frozen, then
applied to the test recordings. Nothing is trained and nothing sees test data
before the thresholds are frozen.

Re-searching the thresholds is required, not optional: averaging logits
changes the score distribution, so the members' own frozen thresholds belong
to different scores.

Output goes to `<window>__<tag>ens<seeds>/seed0/` -- e.g. `w4s_stride1s__L8ens012`
-- in the same `<fold_id>.metrics.json` format, so `evaluate.py` and
`paired_bootstrap.py` read it unchanged. The `seed0` level is only the layout
those scripts expect; `ENSEMBLE.json` beside the metrics says what the run
actually is, and every metrics file repeats it.

    python scripts/ensemble_seeds.py profile=server data=chbmit \\
        model=wearseizure1d_k5only 'train.seeds=[0,1,2]' train.run_tag=L8 \\
        postprocess=hysteresis_widegrid

Folds whose metrics already exist are skipped; `train.force_retrain=true`
re-scores them.
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
from wearseizure.data.splits import load_folds
from wearseizure.models.ensemble import LogitMeanEnsemble, ensemble_tag, missing_member_checkpoints
from wearseizure.models.factory import build_model
from wearseizure.training.engine_baseline import evaluate_fold
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.hashing import sha256_of_file
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir, fold_run_dir, run_tag_from_cfg, seeds_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

torch.multiprocessing.set_sharing_strategy("file_system")

log = get_logger(__name__)


def _load_member(cfg: DictConfig, checkpoint_path: Path) -> torch.nn.Module:
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    seeds = seeds_from_cfg(cfg)
    source_tag = run_tag_from_cfg(cfg)
    out_tag = ensemble_tag(source_tag, seeds)

    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    folds = load_folds(
        str(Path(cfg.split.folds_path)), expected_manifest_hash=hash_manifest(manifest_df)
    )
    member_dirs = [
        fold_run_dir(cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name, cfg.window.name, s, source_tag)
        for s in seeds
    ]
    missing = missing_member_checkpoints(member_dirs, [f.fold_id for f in folds])
    if missing:
        shown = "\n  ".join(str(p) for p in missing[:10])
        raise SystemExit(
            f"{len(missing)} member checkpoint(s) missing, refusing to build a partial ensemble:\n  {shown}"
        )

    out_dir = ensure_dir(
        fold_run_dir(cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name, cfg.window.name, 0, out_tag)
    )
    description = {
        "kind": "inference-time seed ensemble",
        "combine": "mean of member logits",
        "model": cfg.model.name,
        "source_tag": source_tag,
        "member_seeds": seeds,
        "member_dirs": [str(d) for d in member_dirs],
        "note": "the seed0 directory level is layout only; this run has no single training seed",
    }
    (out_dir / "ENSEMBLE.json").write_text(json.dumps(description, indent=2), encoding="utf-8")

    data_dir = cfg.data.generated_dir if cfg.data.name == "synthetic" else None
    raw_dir = cfg.data.raw_dir if cfg.data.name != "synthetic" else None
    records = load_records_from_manifest(manifest_df, data_dir=data_dir, raw_dir=raw_dir)
    threshold_grid = cfg.postprocess.get("threshold_search", OmegaConf.create({}))
    force = bool(cfg.train.get("force_retrain", False))

    n_done = n_skipped = 0
    for fold in folds:
        metrics_path = out_dir / f"{fold.fold_id}.metrics.json"
        if metrics_path.exists() and not force:
            n_skipped += 1
            continue

        member_paths = [d / f"{fold.fold_id}.pt" for d in member_dirs]
        ensemble = LogitMeanEnsemble([_load_member(cfg, p) for p in member_paths])

        result = evaluate_fold(
            model=ensemble,
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

        metrics_path.write_text(
            json.dumps(
                {
                    "fold_id": fold.fold_id,
                    "held_out_key": fold.held_out_key,
                    "ensemble": {
                        **description,
                        "member_checkpoints": [
                            {"path": str(p), "sha256": sha256_of_file(str(p))} for p in member_paths
                        ],
                    },
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
            f"fold {fold.fold_id}: ensemble of seeds {seeds} test sensitivity="
            f"{result.test_event_metrics.sensitivity:.3f} FAR/h={result.test_event_metrics.far_per_hour:.3f}"
        )
        n_done += 1

    log.info(
        f"ensemble {out_tag}: scored {n_done} folds, skipped {n_skipped} already present, "
        f"of {len(folds)} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
