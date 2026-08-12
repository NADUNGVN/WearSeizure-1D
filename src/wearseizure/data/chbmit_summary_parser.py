"""Parser for CHB-MIT's `<subject>-summary.txt` annotation files (server profile).

Standard format (chb01-chb23, which covers all 13 single-channel-eligible
cases in Appendix A):

    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds

Files with more than one seizure use `Seizure N Start/End Time:`; both forms
are handled by the same regex.
"""
from __future__ import annotations

import re
from pathlib import Path

_START_RE = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+(?:\.\d+)?)\s*seconds", re.IGNORECASE)
_END_RE = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+(?:\.\d+)?)\s*seconds", re.IGNORECASE)


def parse_summary_file(path: str) -> dict[str, list[tuple[float, float]]]:
    """Return {edf_filename: [(onset_sec, offset_sec), ...]} for every file
    listed in the summary, including files with zero seizures (empty list).
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = text.split("File Name:")[1:]
    result: dict[str, list[tuple[float, float]]] = {}
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        filename = lines[0].strip()
        starts = [float(m) for m in _START_RE.findall(block)]
        ends = [float(m) for m in _END_RE.findall(block)]
        if len(starts) != len(ends):
            raise ValueError(
                f"{path}: mismatched seizure start/end count for {filename} "
                f"({len(starts)} starts vs {len(ends)} ends)"
            )
        result[filename] = list(zip(starts, ends))
    return result
