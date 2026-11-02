"""The seizure intervals the hardware table scores against.

`hardware_verification_table.py` recovers each seizure from the window labels
rather than from a summary file, so the inversion has to be exact: off by one
second and every delay in the table is wrong by one second while still looking
plausible.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from wearseizure.data.windowing import _overlaps

_spec = importlib.util.spec_from_file_location(
    "hardware_verification_table",
    Path(__file__).resolve().parents[2] / "scripts" / "hardware_verification_table.py",
)
hvt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hvt)


def _rows(seizures, duration_s=600, window_s=4.0):
    rows = []
    for s in range(int(duration_s - window_s) + 1):
        e = s + window_s
        label = int(any(_overlaps(s, e, on, off) for on, off in seizures))
        rows.append({"start_sec": f"{s:.4f}", "end_sec": f"{e:.4f}", "label": str(label)})
    return rows


@pytest.mark.parametrize("seizures", [
    [(100.0, 140.0)],
    [(50.0, 51.0)],
    [(10.0, 70.0), (300.0, 419.0), (500.0, 563.0)],
])
def test_labels_invert_to_the_annotated_intervals(seizures):
    assert hvt.seizures_from_labels(_rows(seizures), 4.0, 1.0) == seizures


def test_refuses_a_stride_it_cannot_invert():
    with pytest.raises(ValueError):
        hvt.seizures_from_labels(_rows([(100.0, 140.0)]), 4.0, 0.5)


def test_whole_recordings_are_found_by_contract_not_by_name():
    found = hvt.whole_recordings(hvt.REPO)
    names = {v.name for _, v in found}
    # chb01_03 lives under hardware/export, the other two under export_<fold>;
    # all three are complete recordings, and no excerpt qualifies.
    assert {"chb01_03_sz0", "chb04_28_sz0", "chb15_40_sz0"} <= names
    assert all(v.parent.name == "whole_recording" for _, v in found)
