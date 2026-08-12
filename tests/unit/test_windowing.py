from __future__ import annotations

import numpy as np
import pytest

from wearseizure.data.synthetic import generate_synthetic_record
from wearseizure.data.windowing import extract, windows_for_edf


def _record_with_seizure():
    rng = np.random.default_rng(0)
    return generate_synthetic_record(
        subject_id="synA", edf_id="synA_00", duration_s=20.0, rng=rng, fs_hz=256.0,
        seizure_intervals=[(8.0, 12.0)],
    )


def test_windows_for_edf_raises_if_edf_not_in_allowed_partition():
    record = _record_with_seizure()
    with pytest.raises(ValueError, match="not in the allowed partition"):
        list(windows_for_edf(record, window_s=4.0, stride_s=1.0, fold_partition_edf_ids={"some_other_edf"}))


def test_windows_never_read_past_their_end_sec():
    record = _record_with_seizure()
    windows = list(
        windows_for_edf(record, window_s=4.0, stride_s=1.0, fold_partition_edf_ids={record.meta.edf_id})
    )
    assert len(windows) > 0
    for w in windows:
        assert w.end_idx <= record.meta.n_samples
        assert w.end_sec <= record.meta.duration_sec + 1e-9
        assert w.n_samples == round(4.0 * record.meta.fs_hz)


def test_window_labels_match_seizure_overlap():
    record = _record_with_seizure()  # seizure at [8.0, 12.0)
    windows = list(
        windows_for_edf(record, window_s=4.0, stride_s=1.0, fold_partition_edf_ids={record.meta.edf_id})
    )
    by_end = {round(w.end_sec, 3): w for w in windows}

    # Window ending well before the seizure starts: no overlap -> label 0.
    assert by_end[6.0].label == 0
    # Window ending inside the seizure (covers [6,10) which overlaps [8,12)): label 1.
    assert by_end[10.0].label == 1
    # Window ending well after the seizure has stopped: label 0.
    assert by_end[20.0].label == 0


def test_extract_returns_correct_slice_and_rejects_mismatched_edf():
    record = _record_with_seizure()
    windows = list(
        windows_for_edf(record, window_s=4.0, stride_s=1.0, fold_partition_edf_ids={record.meta.edf_id})
    )
    w = windows[0]
    segment = extract(record, w)
    assert segment.shape == (w.n_samples,)
    np.testing.assert_array_equal(segment, record.signal[w.start_idx : w.end_idx])

    other = generate_synthetic_record(
        subject_id="synB", edf_id="synB_00", duration_s=5.0, rng=np.random.default_rng(1)
    )
    with pytest.raises(ValueError):
        extract(other, w)
