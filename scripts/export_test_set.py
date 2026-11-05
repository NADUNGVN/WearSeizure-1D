"""Gather the 185 h evaluation test set into one self-contained folder.

The test set is the union of `test_edf_ids` over the 66 folds of
`patient_specific_loso_edf`. It is not stored anywhere on its own; this script
builds it from the raw CHB-MIT tree, the frozen split file and the manifest.

Output layout:
    <out>/
      edf/chbXX/chbXX_YY.edf       full 23-signal EDF (only with --edf)
      npy/chbXX_YY.npy             float32, the one wearable channel, fs=256
      index.csv                    one row per file: channel, seconds, seizures
      folds.json                   copy of the split file used
      README.md                    source, licence, totals

Usage (on SERVER-02):
    export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0
    export WEARSEIZURE_ARTIFACTS_DIR=<artifacts dir used for the runs>
    python scripts/export_test_set.py --out ~/Manh/datasets/WearSeizure-test185h
    # add --edf to also copy the original EDFs (~ several GB)
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    art = os.environ.get("WEARSEIZURE_ARTIFACTS_DIR", "")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=os.environ.get("CHBMIT_RAW_DIR"), help="CHB-MIT root (default: $CHBMIT_RAW_DIR)")
    p.add_argument("--folds", default=f"{art}/splits/patient_specific_loso_edf.json")
    p.add_argument("--manifest", default=f"{art}/manifest/chbmit_manifest.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--edf", action="store_true", help="also copy the original EDF files")
    p.add_argument("--no-npy", action="store_true", help="skip single-channel .npy extraction")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not a.raw_dir:
        raise SystemExit("set --raw-dir or CHBMIT_RAW_DIR")
    raw, out = Path(a.raw_dir).expanduser(), Path(a.out).expanduser()

    folds = json.loads(Path(a.folds).read_text())
    test_ids = sorted({e for f in folds for e in f["test_edf_ids"]})
    if any(e.startswith("syn") for e in test_ids):
        raise SystemExit(f"{a.folds} is the synthetic split, not CHB-MIT")

    with open(a.manifest, newline="") as fh:
        rows = {r["edf_id"]: r for r in csv.DictReader(fh)}
    missing = [e for e in test_ids if e not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} test EDFs not in manifest, e.g. {missing[:3]}")

    out.mkdir(parents=True, exist_ok=True)
    index, total_s, n_sz = [], 0.0, 0
    for i, eid in enumerate(test_ids, 1):
        r = rows[eid]
        src = raw / r["edf_relpath"]
        if not src.exists():  # manifest may store a path relative to the subject dir
            src = raw / r["subject_id"] / Path(r["edf_relpath"]).name
        events = ast.literal_eval(r["seizure_events"]) if r["seizure_events"] else []
        dur = float(r["duration_sec"])
        total_s += dur
        n_sz += len(events)

        if a.edf:
            dst = out / "edf" / r["subject_id"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
        if not a.no_npy:
            import pyedflib

            dst = out / "npy" / f"{eid}.npy"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                rd = pyedflib.EdfReader(str(src))
                try:
                    sig = rd.readSignal(int(r["channel_index_in_edf"])).astype(np.float32)
                finally:
                    rd.close()
                np.save(dst, sig)

        index.append({
            "subject_id": r["subject_id"], "edf_id": eid, "edf_file": src.name,
            "channel_name": r["channel_name"], "channel_index_in_edf": r["channel_index_in_edf"],
            "fs_hz": r["fs_hz"], "n_samples": r["n_samples"], "duration_sec": dur,
            "n_seizures": len(events), "seizure_events_sec": json.dumps(events),
        })
        print(f"[{i}/{len(test_ids)}] {eid}  {dur/3600:.2f} h  {len(events)} sz")

    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    shutil.copy2(a.folds, out / "folds.json")

    hours = total_s / 3600
    subjects = sorted({r["subject_id"] for r in index})
    (out / "README.md").write_text(
        f"# WearSeizure-1D test set ({hours:.1f} h)\n\n"
        f"Union of `test_edf_ids` over {len(folds)} folds (patient_specific_loso_edf).\n"
        f"{len(subjects)} cases, {len(test_ids)} EDF files, {n_sz} seizures, {hours:.1f} h.\n\n"
        "- `npy/<edf_id>.npy`: float32, one wearable channel (see `index.csv`), 256 Hz\n"
        "- `index.csv`: channel, length, seizure onset/offset in seconds\n"
        "- `folds.json`: the split this set was taken from\n\n"
        "Source: CHB-MIT Scalp EEG Database, PhysioNet (https://physionet.org/content/chbmit/1.0.0/),\n"
        "ODC-By 1.0. Cite Shoeb (2009) and Goldberger et al. (2000) when using it.\n"
    )
    print(f"\nDone: {len(test_ids)} files, {n_sz} seizures, {hours:.1f} h -> {out}")
    if abs(hours - 185.0) > 0.5:
        print(f"WARNING: expected 185.0 h, got {hours:.1f} h - check the split/manifest used")


if __name__ == "__main__":
    main()
