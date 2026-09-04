"""Which numeric format? Measure the loss instead of arguing about it.

The question is dynamic fixed point against plain integer, and at what width.
Four cells, two axes:

                 arbitrary scale     power-of-two scale
    16-bit           int16                 dfp16
     8-bit           int8                  dfp8

Plus FP32 as the reference every cell is measured against. The axes are
separated on purpose: "dynamic fixed point instead of INT8" bundles a WIDTH
change with a SCALE-FORMAT change that pull in opposite directions -- widening
helps accuracy, constraining the scale to a power of two hurts it -- and a grid
that moves both at once cannot say which did what.

Two rules that make the measurement mean what it says:

  Calibration uses each fold's VALIDATION partition, never test. A PTQ
  calibrated on test is a leak, and this project exists partly to document that
  practice elsewhere.

  Post-processing thresholds are FROZEN from the FP32 run, read out of its
  metrics.json. Re-fitting them per format would turn the measurement into
  "which format tolerates re-tuning best" rather than "what does this format
  cost".

    python scripts/run_precision_sweep.py profile=server data=chbmit \\
        model=wearseizure1d_k5only train.run_tag=L8 'train.seeds=[0,1,2]'

Writes one JSON per (cell, seed, fold) under
    <artifacts>/precision_sweep/<cell>/<model>[__<tag>]/seed<N>/<fold_id>.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from wearseizure.data.dataset import build_fold_datasets
from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import load_folds
from wearseizure.models.factory import build_model
from wearseizure.quant.ptq import prepare_ptq
from wearseizure.training.engine_baseline import evaluate_fold
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir, fold_run_dir, run_tag_from_cfg, seeds_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.seeding import seed_everything

log = get_logger(__name__)
bootstrap_env(sys.argv)


@dataclass(frozen=True)
class Cell:
    name: str
    bits: int | None          # None = FP32, no quantisation at all
    power_of_two: bool
    per_channel: bool

    @property
    def is_reference(self) -> bool:
        return self.bits is None


# Per-channel is on everywhere it applies: the weight-range measurement put the
# spread at 3.4x, low enough that it is insurance rather than necessity, and it
# is already built. Keeping it constant across cells means the grid measures the
# two axes it is meant to measure and not a third.
CELLS = (
    Cell("fp32", None, False, False),
    Cell("int16", 16, False, True),
    Cell("dfp16", 16, True, True),
    Cell("int8", 8, False, True),
    Cell("dfp8", 8, True, True),
)


def frozen_thresholds(metrics_path: Path) -> dict:
    """The post-processing parameters the FP32 run chose for this fold.

    Read rather than re-derived: re-fitting per format would measure how well
    each format tolerates re-tuning, which is a different question and a
    flattering one.
    """
    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    return d["frozen_postprocess"]["params"]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    run_tag = run_tag_from_cfg(cfg)
    seeds = seeds_from_cfg(cfg)

    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    records = load_records_from_manifest(
        manifest_df,
        data_dir=cfg.data.generated_dir if cfg.data.name == "synthetic" else None,
        raw_dir=cfg.data.raw_dir if cfg.data.name != "synthetic" else None,
    )
    folds = load_folds(str(Path(cfg.split.folds_path)), expected_manifest_hash=hash_manifest(manifest_df))
    if cfg.train.get("max_folds"):
        folds = folds[: cfg.train.max_folds]

    art = Path(cfg.profile.artifacts_dir)
    window_s, stride_s = cfg.window.window_s, cfg.window.stride_s
    n_done = n_skipped = 0

    for seed in seeds:
        src = fold_run_dir(art, cfg.model.name, cfg.split.name, cfg.window.name, seed, tag=run_tag)
        for fold in folds:
            ckpt = Path(src) / f"{fold.fold_id}.pt"
            metrics = Path(src) / f"{fold.fold_id}.metrics.json"
            if not (ckpt.exists() and metrics.exists()):
                raise SystemExit(
                    f"missing {ckpt.name} or its metrics under {src}. This sweep quantises an "
                    "ALREADY TRAINED model and reads the thresholds that run froze; it cannot "
                    "invent either."
                )
            params = frozen_thresholds(metrics)

            for cell in CELLS:
                out_dir = ensure_dir(
                    art / "precision_sweep" / cell.name
                    / (cfg.model.name + (f"__{run_tag}" if run_tag else "")) / f"seed{seed}"
                )
                out_path = out_dir / f"{fold.fold_id}.json"
                if out_path.exists():
                    n_skipped += 1
                    continue

                seed_everything(seed)
                model = build_model(cfg)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)

                if not cell.is_reference:
                    # Calibration data is the fold's VAL partition. Building the
                    # datasets here rather than inside evaluate_fold costs one
                    # extra pass and keeps the calibration source explicit.
                    datasets, _band, _norm = build_fold_datasets(records, fold, window_s, stride_s)
                    cal = DataLoader(
                        datasets["val"], batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.profile.get("num_workers", 0),
                    )
                    model = prepare_ptq(
                        model, cal, weight_bits=cell.bits, act_bits=cell.bits,
                        device=cfg.profile.device,
                        weight_per_channel=cell.per_channel, power_of_two=cell.power_of_two,
                    )

                result = evaluate_fold(
                    model=model, records=records, fold=fold, window_s=window_s, stride_s=stride_s,
                    postprocess_method=params["method"],
                    postprocess_ema_alpha=params["ema_alpha"],
                    postprocess_run_length=params["run_length"],
                    postprocess_event_merge_gap_s=params["event_merge_gap_s"],
                    # Single-element grids: the search is not re-run, the FP32
                    # choice is imposed.
                    threshold_on_grid=[params["threshold_on"]],
                    threshold_off_grid=[params["threshold_off"]],
                    batch_size=cfg.train.batch_size, device=cfg.profile.device,
                    num_workers=cfg.profile.get("num_workers", 0),
                    postprocess_alarm_timestamp=params.get("alarm_timestamp", "window_end"),
                )
                payload = {
                    "cell": cell.name,
                    "bits": cell.bits, "power_of_two": cell.power_of_two,
                    "per_channel": cell.per_channel,
                    "fold_id": fold.fold_id, "seed": seed,
                    "calibrated_on": "val", "thresholds_frozen_from": "fp32",
                    "event": asdict(result.test_event_metrics),
                    "segment": asdict(result.test_segment_metrics),
                }
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                n_done += 1
                log.info(f"{cell.name:<6} seed{seed} {fold.fold_id}: wrote {out_path.name}")

    log.info(f"{n_done} cells computed, {n_skipped} already present")
    log.info("Summarise with: python scripts/summarise_precision_sweep.py $WEARSEIZURE_ARTIFACTS_DIR")


if __name__ == "__main__":
    main()
