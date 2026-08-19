"""Real CHB-MIT EDF loading (server profile only -- unused with synthetic data).

Channel order inside CHB-MIT EDF files is not consistent across recordings,
so the channel is always located by name via the Appendix A channel map
(`manifest.CHBMIT_CHANNEL_MAP` / `subject_to_channel`), never by a fixed index
assumption.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wearseizure.data.manifest import EEGRecordMeta, SeizureEvent, subject_to_channel
from wearseizure.data.records import EEGRecord
from wearseizure.utils.hashing import sha256_of_file


def _find_channel_index(edf_labels: list[str], channel_name: str) -> int:
    normalized = [lbl.strip().upper() for lbl in edf_labels]
    target = channel_name.strip().upper()
    if target in normalized:
        return normalized.index(target)
    raise KeyError(f"channel {channel_name!r} not found among EDF labels {edf_labels}")


def edf_channel_labels(edf_path: str) -> list[str]:
    """Signal labels of an EDF, without reading any sample data.

    Used when building the lever-L5 pre-training corpus: cases outside the 13
    evaluation cases have no Appendix-A position assigned, so which of the four
    wearable positions they actually carry has to be read off the file.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        return list(reader.getSignalLabels())
    finally:
        reader.close()


def load_edf_record(
    edf_path: str,
    subject_id: str,
    edf_id: str,
    seizure_events: list[tuple[float, float]],
    annotation_source: str = "public_chbmit",
    channel_name: str | None = None,
) -> EEGRecord:
    """Load a single channel (per Appendix A) from a real CHB-MIT EDF file.

    Requires `pyedflib`, only needed on the server profile where real data
    lives; kept as a local import so the local/synthetic profile never needs
    the dependency to be functional beyond installation.
    """
    import pyedflib

    # `channel_name=None` keeps the original behaviour: look the position up in
    # the Appendix A map. It is passed explicitly for pre-training-only cases,
    # which are not in that map at all (see manifest.is_evaluation_case), and by
    # `loader.load_records_from_manifest`, which reads it back off the manifest
    # row that recorded it.
    if channel_name is None:
        channel_name = subject_to_channel(subject_id)
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = reader.getSignalLabels()
        ch_idx = _find_channel_index(labels, channel_name)
        fs_hz = reader.getSampleFrequency(ch_idx)
        signal = reader.readSignal(ch_idx).astype(np.float32)
    finally:
        reader.close()

    events = tuple(
        SeizureEvent(event_id=f"{edf_id}_ev{i}", onset_sec=onset, offset_sec=offset)
        for i, (onset, offset) in enumerate(sorted(seizure_events))
    )
    meta = EEGRecordMeta(
        subject_id=subject_id,
        edf_id=edf_id,
        edf_relpath=str(Path(edf_path).name),
        channel_name=channel_name,
        channel_index_in_edf=ch_idx,
        fs_hz=float(fs_hz),
        n_samples=len(signal),
        duration_sec=len(signal) / float(fs_hz),
        annotation_source=annotation_source,
        seizure_events=events,
        raw_sha256=sha256_of_file(edf_path),
    )
    return EEGRecord(meta=meta, signal=signal)
