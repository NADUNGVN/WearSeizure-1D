"""Build (server/chbmit) or validate (local/synthetic) the data manifest.

For real CHB-MIT data, scans each of the 13 Appendix-A subjects' EDF files
against their `<subject>-summary.txt` annotation file. For synthetic data,
the manifest is produced by `generate_synthetic_dataset.py`; this script only
validates it exists and reports its hash, so the documented pipeline order
(make_manifest -> make_splits -> train -> evaluate) is identical across
profiles.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from wearseizure.data.chbmit_summary_parser import parse_summary_file
from wearseizure.data.io_edf import load_edf_record
from wearseizure.data.manifest import (
    CHBMIT_CHANNEL_MAP,
    build_manifest,
    hash_manifest,
    load_manifest,
    save_manifest,
)
from wearseizure.utils.logging import get_logger
from wearseizure.utils.profile_guard import check_profile_data_pairing
from wearseizure.utils.paths import ensure_dir

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


if __name__ == "__main__":
    main()
