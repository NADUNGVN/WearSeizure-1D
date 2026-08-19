"""Load full `EEGRecord`s for every row in a manifest, dispatching between the
synthetic `.npy` cache and real EDF files transparently -- `train.py` /
`evaluate.py` never need to know which profile they are running under.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wearseizure.data.io_edf import load_edf_record
from wearseizure.data.manifest import EEGRecordMeta, events_for_row
from wearseizure.data.records import EEGRecord


def load_records_from_manifest(
    manifest_df: pd.DataFrame,
    data_dir: str | None = None,
    raw_dir: str | None = None,
) -> dict[str, EEGRecord]:
    records: dict[str, EEGRecord] = {}
    for _, row in manifest_df.iterrows():
        events = events_for_row(row)
        if row["annotation_source"] == "synthetic":
            if data_dir is None:
                raise ValueError("data_dir is required to load synthetic records")
            signal = np.load(Path(data_dir) / f"{row['edf_id']}.npy").astype(np.float32)
            meta = EEGRecordMeta(
                subject_id=row["subject_id"],
                edf_id=row["edf_id"],
                edf_relpath=row["edf_relpath"],
                channel_name=row["channel_name"],
                channel_index_in_edf=int(row["channel_index_in_edf"]),
                fs_hz=float(row["fs_hz"]),
                n_samples=int(row["n_samples"]),
                duration_sec=float(row["duration_sec"]),
                annotation_source=row["annotation_source"],
                seizure_events=tuple(events),
                raw_sha256=row.get("raw_sha256", "") or "",
            )
            records[row["edf_id"]] = EEGRecord(meta=meta, signal=signal)
        else:
            if raw_dir is None:
                raise ValueError("raw_dir is required to load real EDF records")
            edf_path = Path(raw_dir) / row["subject_id"] / row["edf_relpath"]
            records[row["edf_id"]] = load_edf_record(
                edf_path=str(edf_path),
                subject_id=row["subject_id"],
                edf_id=row["edf_id"],
                seizure_events=[(e.onset_sec, e.offset_sec) for e in events],
                annotation_source=row["annotation_source"],
                # Take the channel from the manifest row rather than
                # re-deriving it. For the 13 evaluation cases this is the same
                # Appendix A position that was written there in the first
                # place; for lever-L5 pre-training-only cases there is no
                # Appendix A entry to re-derive, and the same EDF may appear
                # once per wearable position.
                channel_name=row["channel_name"],
            )
    return records
