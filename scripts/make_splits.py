"""Generate leakage-safe folds from the manifest (memo 5.1) and freeze them to disk."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import (
    make_patient_specific_loso_edf,
    make_zero_shot_loso_subject,
    save_folds,
)
from wearseizure.utils.logging import get_logger
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.paths import ensure_dir

log = get_logger(__name__)

STRATEGIES = {
    "patient_specific_loso_edf": make_patient_specific_loso_edf,
    "zero_shot_loso_subject": make_zero_shot_loso_subject,
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    manifest_path = Path(cfg.data.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found - run make_manifest.py first")
    manifest_df = load_manifest(str(manifest_path))

    strategy_fn = STRATEGIES.get(cfg.split.strategy)
    if strategy_fn is None:
        raise ValueError(f"unknown split strategy {cfg.split.strategy!r}, expected one of {list(STRATEGIES)}")

    folds = strategy_fn(manifest_df, seed=cfg.seed, val_fraction=cfg.split.val_fraction)
    if not folds:
        raise RuntimeError("split strategy produced zero folds - check the manifest")

    folds_path = Path(cfg.split.folds_path)
    ensure_dir(folds_path.parent)
    save_folds(folds, str(folds_path))
    log.info(
        f"Wrote {len(folds)} folds ({cfg.split.strategy}) -> {folds_path}, "
        f"manifest_hash={hash_manifest(manifest_df)}"
    )


if __name__ == "__main__":
    main()
