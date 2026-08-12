from __future__ import annotations

import pytest

from wearseizure.data.manifest import (
    CHBMIT_CHANNEL_MAP,
    EEGRecordMeta,
    SeizureEvent,
    build_manifest,
    hash_manifest,
    load_manifest,
    save_manifest,
    subject_to_channel,
)


def _meta(edf_id: str, events=()) -> EEGRecordMeta:
    return EEGRecordMeta(
        subject_id="synA",
        edf_id=edf_id,
        edf_relpath=f"{edf_id}.npy",
        channel_name="SYN-CH1",
        channel_index_in_edf=0,
        fs_hz=256.0,
        n_samples=256 * 60,
        duration_sec=60.0,
        annotation_source="synthetic",
        seizure_events=tuple(events),
    )


def test_subject_to_channel_matches_appendix_a():
    assert subject_to_channel("chb03") == "Fp1-F3"
    assert subject_to_channel("chb02") == "P7-O1"
    assert subject_to_channel("chb01") == "P8-O2"
    with pytest.raises(KeyError):
        subject_to_channel("chb99")


def test_appendix_a_case_counts():
    assert len(CHBMIT_CHANNEL_MAP["Fp1-F3"]) == 5
    assert len(CHBMIT_CHANNEL_MAP["P7-O1"]) == 5
    assert len(CHBMIT_CHANNEL_MAP["P8-O2"]) == 3
    assert len(CHBMIT_CHANNEL_MAP["Fp2-F4"]) == 0


def test_build_manifest_rejects_duplicate_edf_id():
    metas = [_meta("edfA"), _meta("edfA")]
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest(metas)


def test_build_manifest_rejects_unknown_annotation_source():
    bad = EEGRecordMeta(
        subject_id="synA",
        edf_id="edfX",
        edf_relpath="edfX.npy",
        channel_name="SYN-CH1",
        channel_index_in_edf=0,
        fs_hz=256.0,
        n_samples=256,
        duration_sec=1.0,
        annotation_source="not_a_real_source",
    )
    with pytest.raises(ValueError, match="annotation_source"):
        build_manifest([bad])


def test_seizure_event_requires_positive_duration():
    with pytest.raises(ValueError):
        SeizureEvent(event_id="e1", onset_sec=10.0, offset_sec=5.0)


def test_manifest_roundtrip_and_hash_stable(tmp_path):
    events = [SeizureEvent(event_id="e0", onset_sec=10.0, offset_sec=20.0)]
    metas = [_meta("edfA", events=events), _meta("edfB")]
    df = build_manifest(metas)

    path = tmp_path / "manifest.csv"
    save_manifest(df, str(path))
    loaded = load_manifest(str(path))

    assert hash_manifest(df) == hash_manifest(loaded)
    assert loaded.loc[loaded.edf_id == "edfA", "seizure_events"].iloc[0] == [
        {"event_id": "e0", "onset_sec": 10.0, "offset_sec": 20.0}
    ]


def test_manifest_hash_changes_with_content():
    df1 = build_manifest([_meta("edfA")])
    df2 = build_manifest([_meta("edfB")])
    assert hash_manifest(df1) != hash_manifest(df2)

    df1_again = build_manifest([_meta("edfA")])
    assert hash_manifest(df1) == hash_manifest(df1_again)
