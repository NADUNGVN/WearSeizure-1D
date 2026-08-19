"""Build (server/chbmit) or validate (local/synthetic) the data manifest.

For real CHB-MIT data, scans each of the 13 Appendix-A subjects' EDF files
against their `<subject>-summary.txt` annotation file. For synthetic data,
the manifest is produced by `generate_synthetic_dataset.py`; this script only
validates it exists and reports its hash, so the documented pipeline order
(make_manifest -> make_splits -> train -> evaluate) is identical across
profiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from wearseizure.data.chbmit_summary_parser import parse_summary_file
from wearseizure.data.io_edf import edf_channel_labels, load_edf_record
from wearseizure.data.manifest import (
    CHBMIT_CHANNEL_MAP,
    CHBMIT_WEARABLE_CHANNELS,
    build_manifest,
    hash_manifest,
    is_evaluation_case,
    load_manifest,
    save_manifest,
)
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir
from wearseizure.utils.profile_guard import check_profile_data_pairing

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

log = get_logger(__name__)


def _build_chbmit_manifest(raw_dir: str, annotation_source: str):
    records = []
    for subjects in CHBMIT_CHANNEL_MAP.values():
        for subject_id in subjects:
            subject_dir = Path(raw_dir) / subject_id
            summary_path = subject_dir / f"{subject_id}-summary.txt"
            if not summary_path.exists():
                raise FileNotFoundError(f"missing summary file for {subject_id}: {summary_path}")
            events_by_file = parse_summary_file(str(summary_path))
            for edf_filename, events in events_by_file.items():
                edf_path = subject_dir / edf_filename
                if not edf_path.exists():
                    log.warning(f"{edf_path} listed in summary but missing on disk, skipping")
                    continue
                record = load_edf_record(
                    edf_path=str(edf_path),
                    subject_id=subject_id,
                    edf_id=edf_path.stem,
                    seizure_events=events,
                    annotation_source=annotation_source,
                )
                records.append(record.meta)
    if not records:
        raise RuntimeError(f"no EDF records found under {raw_dir}")
    return build_manifest(records)


def _pretrain_only_subjects(raw_dir: str) -> list[str]:
    """CHB-MIT cases present on disk that the protocol does NOT evaluate on.

    Discovered by scanning rather than hardcoded, so this cannot silently drift
    from whatever the server actually has: a case counts only if its
    `<case>-summary.txt` is there to be parsed.
    """
    root = Path(raw_dir)
    found = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / f"{d.name}-summary.txt").exists()
    )
    return [s for s in found if not is_evaluation_case(s)]


def _build_chbmit_pretrain_manifest(raw_dir: str, annotation_source: str, channels):
    """Lever L5: a manifest of the NON-evaluation CHB-MIT cases, one row per
    (EDF, wearable position).

    Kept in a separate file from the evaluation manifest on purpose. Merging
    them would put these cases into `data/splits.py` and change what the
    protocol evaluates -- the whole point is that evaluation stays at the 13
    Appendix-A cases while pre-training gets more data.
    """
    channels = [c for c in channels if c in CHBMIT_WEARABLE_CHANNELS] or list(CHBMIT_WEARABLE_CHANNELS)
    subjects = _pretrain_only_subjects(raw_dir)
    if not subjects:
        raise RuntimeError(
            f"no non-evaluation CHB-MIT cases found under {raw_dir}; lever L5 has nothing to add"
        )
    log.info(f"pre-training-only cases found: {subjects}")

    records = []
    skipped: dict[str, int] = {}
    for subject_id in subjects:
        subject_dir = Path(raw_dir) / subject_id
        events_by_file = parse_summary_file(str(subject_dir / f"{subject_id}-summary.txt"))
        for edf_filename, events in events_by_file.items():
            edf_path = subject_dir / edf_filename
            if not edf_path.exists():
                log.warning(f"{edf_path} listed in summary but missing on disk, skipping")
                continue
            labels = {lbl.strip().upper() for lbl in edf_channel_labels(str(edf_path))}
            for channel in channels:
                if channel.upper() not in labels:
                    skipped[channel] = skipped.get(channel, 0) + 1
                    continue
                record = load_edf_record(
                    edf_path=str(edf_path),
                    subject_id=subject_id,
                    # "@<channel>" and not "__": `data/splits.subject_from_fold_id`
                    # splits fold ids on "__", and while pre-training records
                    # never become folds, an id that could be mistaken for one
                    # is a trap not worth leaving.
                    edf_id=f"{edf_path.stem}@{channel}",
                    seizure_events=events,
                    annotation_source=annotation_source,
                    channel_name=channel,
                )
                records.append(record.meta)
    if skipped:
        log.warning(f"positions missing from some EDFs (rows skipped): {skipped}")
    if not records:
        raise RuntimeError(f"no pre-training records built from {raw_dir}")
    return build_manifest(records)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    manifest_path = Path(cfg.data.manifest_path)

    if cfg.data.name == "synthetic":
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. For synthetic data, run "
                "scripts/generate_synthetic_dataset.py first."
            )
        df = load_manifest(str(manifest_path))
        log.info(f"Validated existing synthetic manifest: {len(df)} rows, hash={hash_manifest(df)}")
        return

    ensure_dir(manifest_path.parent)
    df = _build_chbmit_manifest(cfg.data.raw_dir, cfg.data.annotation_source)
    save_manifest(df, str(manifest_path))
    log.info(f"Wrote CHB-MIT manifest: {len(df)} rows -> {manifest_path}, hash={hash_manifest(df)}")

    pretrain_path = cfg.data.get("pretrain_manifest_path")
    if pretrain_path:
        pretrain_path = Path(pretrain_path)
        ensure_dir(pretrain_path.parent)
        pdf = _build_chbmit_pretrain_manifest(
            cfg.data.raw_dir,
            cfg.data.annotation_source,
            list(cfg.data.get("pretrain_channels", CHBMIT_WEARABLE_CHANNELS)),
        )
        save_manifest(pdf, str(pretrain_path))
        log.info(
            f"Wrote CHB-MIT lever-L5 pre-training manifest: {len(pdf)} rows "
            f"({pdf['subject_id'].nunique()} non-evaluation cases, "
            f"{pdf['duration_sec'].sum() / 3600:.1f}h of single-channel signal) "
            f"-> {pretrain_path}, hash={hash_manifest(pdf)}"
        )


if __name__ == "__main__":
    main()
