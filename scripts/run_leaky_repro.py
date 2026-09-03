"""Item A7 -- reproduce the published protocol on our own code and data.

The claim this exists to turn into a measurement: the gap between the published
0.9962 and our reproduction's 0.8811 is caused by the evaluation PROTOCOL, not
by our implementation being worse. Nothing on record rules out the second
reading, and a reviewer will say so.

Design
------
Every rung uses the SAME 66 folds and the SAME pooled recordings. Only the rule
that partitions them changes, so a difference between rungs cannot come from
different data. Rung D is this project's own protocol, which means it must
reproduce the numbers already in the experiment log -- if it does not, this
harness is wrong and nothing else it reports can be trusted.

No cohort pre-training anywhere here: the published work did not do it, and the
number being reproduced is a from-scratch one.

    python scripts/run_leaky_repro.py profile=server data=chbmit \\
        model=baseline_frontiers2d +rung=A_as_published

Writes one JSON per fold under
    <artifacts>/leaky_repro/<rung>/<model>/<fold_id>.json
so two servers can shard by rung or model without colliding on a shared disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from wearseizure.data.dataset import WearSeizureWindowDataset
from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import load_folds
from wearseizure.eval.leaky_protocol import (
    LADDER,
    near_duplicate_fraction,
    prepare_records,
    split_window_indices,
)
from wearseizure.eval.metrics_segment import compute_segment_metrics
from wearseizure.models.factory import build_model
from wearseizure.training.loop import train_classifier
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.seeding import seed_everything

log = get_logger(__name__)
bootstrap_env(sys.argv)
# NOT set_sharing_strategy("file_system") here, unlike scripts/train.py.
#
# That strategy spawns torch_shm_manager, which needs to create a socket
# directory under the system temp dir. On SERVER-04 it cannot, and every job
# died with "could not generate a random directory for manager socket" -- a
# host difference, nothing to do with the data or the model. train.py needs the
# strategy because it opens DataLoaders across 66 folds and exhausts file
# descriptors otherwise; this script does not, because the workload is
# kernel-launch bound and runs with few workers or none at all.

RUNGS = {c.name: c for c in LADDER}


def _score(model, dataset, indices, device, batch_size, num_workers):
    """Positive-class probabilities for `indices`, in that order."""
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    model = model.to(device).eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            scores.append(p.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def _best_threshold(labels, scores, grid) -> float:
    """The threshold maximising balanced accuracy on whatever is passed in.

    Passed the TEST partition, this is the fitting leak; passed the VAL
    partition, it is the ordinary honest choice. The function is deliberately
    the same one in both arms so the rungs differ only in what they are shown.
    """
    best, best_score = grid[0], -1.0
    for t in grid:
        m = compute_segment_metrics(labels, scores, threshold=t)
        if m.balanced_accuracy > best_score:
            best, best_score = t, m.balanced_accuracy
    return best


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    rung_name = cfg.get("rung")
    if rung_name not in RUNGS:
        raise SystemExit(f"+rung= must be one of {sorted(RUNGS)}, got {rung_name!r}")
    rung = RUNGS[rung_name]
    if not rung.segment_metric:
        raise SystemExit(
            f"{rung.name} is the event-level rung, which is this project's own protocol -- "
            "its numbers come from scripts/train.py and the experiment log, not from here."
        )
    log.info(rung.describe())

    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    # Same source resolution as scripts/train.py, so this harness can be smoke
    # tested on the synthetic cohort without touching the clinical data.
    records = load_records_from_manifest(
        manifest_df,
        data_dir=cfg.data.generated_dir if cfg.data.name == "synthetic" else None,
        raw_dir=cfg.data.raw_dir if cfg.data.name != "synthetic" else None,
    )
    folds = load_folds(str(Path(cfg.split.folds_path)), expected_manifest_hash=hash_manifest(manifest_df))
    max_folds = cfg.train.get("max_folds")
    if max_folds:
        folds = folds[:max_folds]

    out_dir = ensure_dir(Path(cfg.profile.artifacts_dir) / "leaky_repro" / rung.name / cfg.model.name)
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    seed = int(cfg.seed)

    for fold in folds:
        out_path = out_dir / f"{fold.fold_id}.json"
        if out_path.exists():
            log.info(f"{fold.fold_id}: already done, skipping")
            continue
        seed_everything(seed)

        pool = fold.train_edf_ids | fold.val_edf_ids | fold.test_edf_ids
        prepared = prepare_records(
            records, pool, fold.train_edf_ids, global_normalisation=rung.global_normalisation
        )
        dataset = WearSeizureWindowDataset(
            prepared, frozenset(pool), cfg.window.window_s, cfg.window.stride_s
        )
        train_idx, val_idx, test_idx = split_window_indices(
            dataset,
            random_window_split=rung.random_window_split,
            test_edf_ids=fold.test_edf_ids,
            val_fraction=cfg.split.val_fraction,
            seed=seed,
        )
        overlap = near_duplicate_fraction(dataset, train_idx, test_idx)
        log.info(
            f"{fold.fold_id}: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test "
            f"windows; {overlap:.1%} of test windows share samples with a training window"
        )

        dl = {"batch_size": cfg.train.batch_size, "num_workers": cfg.profile.get("num_workers", 0)}
        model = build_model(cfg)
        result = train_classifier(
            model,
            DataLoader(Subset(dataset, train_idx.tolist()), shuffle=True, **dl),
            DataLoader(Subset(dataset, val_idx.tolist()), shuffle=False, **dl),
            epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
            device=cfg.profile.device,
            early_stopping_patience=cfg.train.early_stopping_patience,
        )

        test_scores, test_labels = _score(
            result.model, dataset, test_idx, cfg.profile.device, cfg.train.batch_size, dl["num_workers"]
        )
        if rung.threshold_on_test:
            threshold = _best_threshold(test_labels, test_scores, grid)
        else:
            val_scores, val_labels = _score(
                result.model, dataset, val_idx, cfg.profile.device, cfg.train.batch_size, dl["num_workers"]
            )
            threshold = _best_threshold(val_labels, val_scores, grid)

        m = compute_segment_metrics(test_labels, test_scores, threshold=threshold)
        payload = {
            "rung": rung.name,
            "rung_description": rung.describe(),
            "model": cfg.model.name,
            "fold_id": fold.fold_id,
            "seed": seed,
            "n_train_windows": len(train_idx),
            "n_val_windows": len(val_idx),
            "n_test_windows": len(test_idx),
            "test_window_overlap_fraction": float(overlap),
            "threshold": float(threshold),
            "threshold_fitted_on": "test" if rung.threshold_on_test else "val",
            "segment": {
                "sensitivity": m.sensitivity, "specificity": m.specificity,
                "f1": m.f1, "accuracy": m.accuracy, "balanced_accuracy": m.balanced_accuracy,
                "auprc": m.auprc, "auroc": m.auroc, "prevalence": m.prevalence,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info(
            f"{fold.fold_id}: window sensitivity={m.sensitivity:.4f} specificity={m.specificity:.4f} "
            f"balanced_acc={m.balanced_accuracy:.4f} -> {out_path}"
        )

    log.info(f"rung {rung.name}, model {cfg.model.name}: written to {out_dir}")


if __name__ == "__main__":
    main()
