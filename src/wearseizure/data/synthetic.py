"""Synthetic EEG generator: same manifest schema and `EEGRecord` shape as real
CHB-MIT data, so the entire pipeline can be exercised on a laptop CPU without
any clinical data. Not physiologically validated -- it exists purely so unit
and integration tests exercise real data shapes and timing, never to produce
numbers that mean anything clinically (see `enforce_gates=false` under
`profile=local_synthetic`).
"""
from __future__ import annotations

import numpy as np

from wearseizure.data.manifest import EEGRecordMeta, SeizureEvent, build_manifest
from wearseizure.data.records import EEGRecord
from wearseizure.signal.filters import CausalBandpass

SYNTHETIC_CHANNEL_NAME = "SYN-CH1"


def _ictal_envelope(t: np.ndarray, onset: float, offset: float, rise_fall: float = 1.0) -> np.ndarray:
    rise = np.clip((t - onset) / rise_fall, 0.0, 1.0)
    fall = np.clip((offset - t) / rise_fall, 0.0, 1.0)
    return np.minimum(rise, fall).clip(0.0, 1.0)


def generate_synthetic_record(
    subject_id: str,
    edf_id: str,
    duration_s: float,
    rng: np.random.Generator,
    fs_hz: float = 256.0,
    seizure_intervals: list[tuple[float, float]] | None = None,
    background_amplitude: float = 20.0,
    ictal_amplitude: float = 80.0,
    artifact_rate_per_hour: float = 6.0,
    artifact_amplitude: float = 300.0,
    annotation_source: str = "synthetic",
) -> EEGRecord:
    seizure_intervals = seizure_intervals or []
    n_samples = round(duration_s * fs_hz)
    t = np.arange(n_samples) / fs_hz

    background = rng.standard_normal(n_samples) * background_amplitude
    background = CausalBandpass(fs_hz=fs_hz).apply_full_reset(background)

    signal = background.copy()
    for onset, offset in seizure_intervals:
        env = _ictal_envelope(t, onset, offset) * ictal_amplitude
        freqs = rng.uniform(3.0, 20.0, size=3)
        phases = rng.uniform(0, 2 * np.pi, size=3)
        ictal_wave = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases))
        signal = signal + env * ictal_wave / len(freqs)

    n_artifacts = rng.poisson(artifact_rate_per_hour * duration_s / 3600.0)
    for _ in range(n_artifacts):
        idx = rng.integers(0, n_samples)
        width = rng.integers(1, int(0.2 * fs_hz) + 1)
        end = min(idx + width, n_samples)
        signal[idx:end] += rng.choice([-1.0, 1.0]) * artifact_amplitude

    events = [
        SeizureEvent(event_id=f"{edf_id}_ev{i}", onset_sec=onset, offset_sec=offset)
        for i, (onset, offset) in enumerate(seizure_intervals)
    ]
    meta = EEGRecordMeta(
        subject_id=subject_id,
        edf_id=edf_id,
        edf_relpath=f"synthetic/{edf_id}.npy",
        channel_name=SYNTHETIC_CHANNEL_NAME,
        channel_index_in_edf=0,
        fs_hz=fs_hz,
        n_samples=n_samples,
        duration_sec=duration_s,
        annotation_source=annotation_source,
        seizure_events=tuple(events),
    )
    return EEGRecord(meta=meta, signal=signal.astype(np.float32))


def generate_synthetic_cohort(
    n_subjects: int,
    edfs_per_subject: int,
    seed: int,
    edf_duration_s: float = 600.0,
    interictal_edf_fraction: float = 0.4,
    max_seizures_per_edf: int = 2,
):
    """Build a small synthetic cohort with the same shape as a real manifest:
    some EDFs interictal-only, others containing 1+ seizures, spread across
    several subjects -- enough to exercise both split strategies.

    Seizure onset/duration are sized proportionally to `edf_duration_s`
    (rather than fixed absolute margins) so short EDFs used in fast unit
    tests don't produce an invalid (empty or negative) sampling range.
    """
    if edf_duration_s < 10.0:
        raise ValueError(f"edf_duration_s must be >= 10.0 to fit a seizure, got {edf_duration_s}")

    rng = np.random.default_rng(seed)
    records: dict[str, EEGRecord] = {}
    all_meta: list[EEGRecordMeta] = []

    onset_low = edf_duration_s * 0.15
    onset_high = edf_duration_s * 0.60
    duration_low = max(1.0, edf_duration_s * 0.05)
    duration_high = max(duration_low + 1.0, edf_duration_s * 0.20)

    for s in range(n_subjects):
        subject_id = f"syn{s:02d}"
        n_interictal = max(1, round(edfs_per_subject * interictal_edf_fraction))
        for e in range(edfs_per_subject):
            edf_id = f"{subject_id}_{e:02d}"
            is_interictal = e < n_interictal
            seizure_intervals: list[tuple[float, float]] = []
            if not is_interictal:
                n_sz = rng.integers(1, max_seizures_per_edf + 1)
                for _ in range(n_sz):
                    onset = float(rng.uniform(onset_low, onset_high))
                    duration = float(rng.uniform(duration_low, duration_high))
                    seizure_intervals.append((onset, min(onset + duration, edf_duration_s * 0.98)))
                seizure_intervals.sort()
            record = generate_synthetic_record(
                subject_id=subject_id,
                edf_id=edf_id,
                duration_s=edf_duration_s,
                rng=rng,
                seizure_intervals=seizure_intervals,
            )
            records[edf_id] = record
            all_meta.append(record.meta)

    manifest_df = build_manifest(all_meta)
    return manifest_df, records
