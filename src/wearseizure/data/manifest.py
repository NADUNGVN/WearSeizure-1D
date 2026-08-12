"""Manifest schema and (de)serialization.

The manifest is the single source of truth for what data exists and how it is
labeled. One row = one EDF recording. Splitting (``data/splits.py``) always
operates on manifest rows keyed by ``edf_id`` / ``subject_id`` -- never on
individual windows -- which is what makes the leakage-safe protocol possible
(Research Decision Memo, section 2.2 / 5.1).

Stored on disk as CSV with ``seizure_events`` JSON-encoded in a single column,
so the manifest stays a single flat, diffable, human-inspectable file while
still carrying per-event onset/offset lists.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from wearseizure.utils.hashing import canonical_json, sha256_of

MANIFEST_VERSION = 1

# Appendix A of the Research Decision Memo: the four wearable channel
# positions and the CHB-MIT cases whose seizure onset was clinically
# confirmed to be observable from that channel (Chung et al. 2024).
CHBMIT_CHANNEL_MAP: dict[str, list[str]] = {
    "Fp1-F3": ["chb03", "chb07", "chb08", "chb22", "chb23"],
    "P7-O1": ["chb02", "chb05", "chb10", "chb11", "chb15"],
    "P8-O2": ["chb01", "chb04", "chb17"],
    "Fp2-F4": [],
}


def subject_to_channel(subject_id: str) -> str:
    for channel, subjects in CHBMIT_CHANNEL_MAP.items():
        if subject_id in subjects:
            return channel
    raise KeyError(
        f"{subject_id!r} is not one of the 13 single-channel-eligible CHB-MIT cases"
    )


@dataclass(frozen=True)
class SeizureEvent:
    event_id: str
    onset_sec: float
    offset_sec: float

    def __post_init__(self) -> None:
        if self.offset_sec <= self.onset_sec:
            raise ValueError(
                f"event {self.event_id}: offset_sec must be > onset_sec "
                f"({self.offset_sec} <= {self.onset_sec})"
            )


@dataclass(frozen=True)
class EEGRecordMeta:
    subject_id: str
    edf_id: str
    edf_relpath: str
    channel_name: str
    channel_index_in_edf: int
    fs_hz: float
    n_samples: int
    duration_sec: float
    annotation_source: str  # "public_chbmit" | "frontiers_reannotation" | "synthetic"
    seizure_events: tuple[SeizureEvent, ...] = field(default_factory=tuple)
    raw_sha256: str = ""
    manifest_version: int = MANIFEST_VERSION

    @property
    def is_interictal_only(self) -> bool:
        return len(self.seizure_events) == 0

    def to_row(self) -> dict:
        d = asdict(self)
        d["seizure_events"] = canonical_json(
            [{"event_id": e.event_id, "onset_sec": e.onset_sec, "offset_sec": e.offset_sec} for e in self.seizure_events]
        )
        d["is_interictal_only"] = self.is_interictal_only
        return d


VALID_ANNOTATION_SOURCES = {"public_chbmit", "frontiers_reannotation", "synthetic"}


def build_manifest(records: list[EEGRecordMeta]) -> pd.DataFrame:
    if not records:
        raise ValueError("build_manifest: at least one record is required")
    edf_ids = [r.edf_id for r in records]
    if len(set(edf_ids)) != len(edf_ids):
        dupes = {e for e in edf_ids if edf_ids.count(e) > 1}
        raise ValueError(f"build_manifest: duplicate edf_id(s): {dupes}")
    for r in records:
        if r.annotation_source not in VALID_ANNOTATION_SOURCES:
            raise ValueError(
                f"{r.edf_id}: unknown annotation_source {r.annotation_source!r}, "
                f"expected one of {VALID_ANNOTATION_SOURCES}"
            )
    return pd.DataFrame([r.to_row() for r in records])


def save_manifest(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_manifest(path: str) -> pd.DataFrame:
    # keep_default_na=False: an empty raw_sha256 field must round-trip as ""
    # (matching the in-memory manifest before it was ever saved), not as NaN.
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    df["seizure_events"] = df["seizure_events"].apply(json.loads)
    return df


def events_for_row(row: pd.Series) -> list[SeizureEvent]:
    events = row["seizure_events"]
    if isinstance(events, str):
        events = json.loads(events)
    return [SeizureEvent(**e) for e in events]


def hash_manifest(df: pd.DataFrame) -> str:
    """Content hash independent of row/column order, used to version-lock splits.

    Normalizes `seizure_events` to a parsed list-of-dicts regardless of
    whether the DataFrame came straight from `build_manifest` (still a JSON
    string, ready for CSV) or from `load_manifest` (already parsed) -- so the
    hash is identical before and after a save/load round-trip.
    """
    records = df.to_dict(orient="records")
    for r in records:
        if isinstance(r["seizure_events"], str):
            r["seizure_events"] = json.loads(r["seizure_events"])
    records_sorted = sorted(records, key=lambda r: r["edf_id"])
    return sha256_of(records_sorted)
