"""Materialize a synthetic cohort to disk (signals + manifest together).

This is the local-only counterpart to running real CHB-MIT data through
`make_manifest.py` on the server: after this script, the rest of the pipeline
(`make_splits.py`, `train.py`, `evaluate.py`) runs identically regardless of
where the data came from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from wearseizure.data.manifest import save_manifest
from wearseizure.data.synthetic import generate_synthetic_cohort
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.data.name != "synthetic":
        raise ValueError(
            f"generate_synthetic_dataset.py requires data=synthetic, got data={cfg.data.name!r}. "
            "Run with profile=local_synthetic."
        )

    out_dir = ensure_dir(cfg.data.generated_dir)
    manifest_df, records = generate_synthetic_cohort(
        n_subjects=cfg.data.n_subjects,
        edfs_per_subject=cfg.data.edfs_per_subject,
        seed=cfg.seed,
        edf_duration_s=cfg.data.edf_duration_s,
    )

    for edf_id, record in records.items():
        np.save(out_dir / f"{edf_id}.npy", record.signal)

    manifest_path = Path(cfg.data.manifest_path)
    ensure_dir(manifest_path.parent)
    save_manifest(manifest_df, str(manifest_path))

    n_seizures = sum(len(r.meta.seizure_events) for r in records.values())
    log.info(
        f"Generated {len(records)} synthetic EDFs, {n_seizures} seizure events, "
        f"across {cfg.data.n_subjects} subjects -> {out_dir}"
    )
    log.info(f"Manifest written to {manifest_path} ({len(manifest_df)} rows)")


if __name__ == "__main__":
    main()
