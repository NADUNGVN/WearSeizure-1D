"""Build the 13 cohort pre-trained initialisations up front (lever L1).

`scripts/train.py` already builds these lazily inside its fold loop, so this
script is not required -- it exists because building them lazily is strictly
sequential, and on this hardware that wastes most of the machine.

Why sequential is wasteful here
-------------------------------
The model is ~12k parameters / ~0.59M MACs. A batch of 256 windows is a few
hundred MFLOP, which a Quadro RTX 8000 finishes in microseconds; per-iteration
wall-clock is dominated by **kernel-launch overhead** (tens of launches per
forward/backward at ~5-10us each), not by arithmetic. Raising the batch size
fights that only weakly and changes optimisation semantics, so it is not a free
throughput knob. Running several independent pre-trainings **concurrently** on
the same GPU is, because each one leaves the device almost entirely idle.

The 13 pre-trainings are fully independent (one per held-out patient) and each
writes its own cache file, so sharding them across processes is safe with no
locking.

Usage
-----
`data=chbmit` is NOT optional: configs/config.yaml defaults to `data: synthetic`
and the `profile` group does not change it. See utils/profile_guard.py.
Single process, whole machine to itself::

    python scripts/pretrain_cohort.py profile=server data=chbmit profile.num_workers=14

Three concurrent shards (recommended when nothing else is on the box) -- note
the *lower* per-process worker count, so the three processes together stay
under the 14 physical cores::

    for i in 0 1 2; do
      python scripts/pretrain_cohort.py profile=server data=chbmit \
        +shard=$i +n_shards=3 profile.num_workers=4 &
    done; wait

Then run training as normal; it will find every init already cached::

    python scripts/train.py profile=server data=chbmit train.pretrain.enabled=true
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import load_manifest
from wearseizure.models.factory import build_model
from wearseizure.training.pretrain import get_or_train_cohort_init
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.paths import ensure_dir
from wearseizure.utils.seeding import seed_everything

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

# Same rationale as scripts/train.py: many DataLoader worker pools over many
# subjects exhaust file descriptors under the default sharing strategy.
torch.multiprocessing.set_sharing_strategy("file_system")

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    seed_everything(cfg.seed)

    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    data_dir = cfg.data.generated_dir if cfg.data.name == "synthetic" else None
    raw_dir = cfg.data.raw_dir if cfg.data.name != "synthetic" else None
    records = load_records_from_manifest(manifest_df, data_dir=data_dir, raw_dir=raw_dir)

    subjects = sorted(manifest_df["subject_id"].unique())
    shard = int(cfg.get("shard", 0))
    n_shards = int(cfg.get("n_shards", 1))
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard={shard} must be in [0, n_shards={n_shards})")
    mine = subjects[shard::n_shards]

    pretrain_cfg = cfg.train.get("pretrain", {})
    cache_dir = ensure_dir(
        Path(cfg.profile.artifacts_dir) / "pretrain" / cfg.model.name / cfg.window.name
    )
    log.info(
        f"shard {shard}/{n_shards}: pre-training {len(mine)} of {len(subjects)} subjects "
        f"({', '.join(mine)}) -> {cache_dir}"
    )

    for subject in mine:
        get_or_train_cohort_init(
            records=records,
            manifest_df=manifest_df,
            held_out_subject=subject,
            model_factory=lambda: build_model(cfg),
            cache_dir=cache_dir,
            window_s=cfg.window.window_s,
            stride_s=cfg.window.stride_s,
            seed=cfg.seed,
            epochs=pretrain_cfg.get("epochs", cfg.train.epochs),
            lr=pretrain_cfg.get("lr", cfg.train.lr),
            weight_decay=cfg.train.weight_decay,
            batch_size=pretrain_cfg.get("batch_size", cfg.train.batch_size),
            device=cfg.profile.device,
            early_stopping_patience=pretrain_cfg.get(
                "early_stopping_patience", cfg.train.early_stopping_patience
            ),
            num_workers=cfg.profile.get("num_workers", 0),
            val_subject_fraction=pretrain_cfg.get("val_subject_fraction", 0.2),
            class_balanced_sampling=cfg.train.class_balanced_sampling,
            force=pretrain_cfg.get("force", False),
        )

    log.info(f"shard {shard}/{n_shards}: done, {len(mine)} initialisation(s) in {cache_dir}")


if __name__ == "__main__":
    main()
