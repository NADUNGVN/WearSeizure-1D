"""Which folds make the best onset vectors: every seizure caught, nothing else fired.

    python scripts/rank_folds_for_vectors.py $WEARSEIZURE_ARTIFACTS_DIR

Reads the DFP8 evaluation that already ran over all 66 folds and ranks them by
how cleanly the **integer datapath** handled each held-out recording. Nothing is
recomputed and no model is loaded; this is a few hundred JSON files.

**It does not rank by accuracy, on purpose.** Over 4 s windows at 1 s stride a
recording is overwhelmingly non-seizure, so accuracy is a statement about the
majority class: a fold can score 99 % and still miss the seizure entirely. The
project's own headline result is that accuracy cannot separate protocols on this
data. What makes a demonstration honest is that every seizure was caught and
little else fired, so the ranking is on **event sensitivity first, false-alarm
rate second**.

The output is the input to `export_dfp_hardware.py +export.fold=<id>` followed by
`export_onset_vectors.py`: each fold has its own weights under the
patient-specific protocol, so vectors from a fold need that fold exported first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(root: Path, bits: int, seed: int) -> list[dict]:
    base = root / "dfp_eval" / f"dfp{bits}" / f"seed{seed}"
    if not base.is_dir():
        raise SystemExit(
            f"no DFP{bits} evaluation under {base}. Run export_dfp_hardware.py with "
            "+export.evaluate=true first, or point --bits/--seed at an arm that exists."
        )
    rows = []
    for path in sorted(base.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("refit_thresholds"):
            # Re-fitted thresholds measure a different question and score
            # differently. Mixing the arms would rank folds on which arm they
            # happened to come from.
            continue
        rows.append(d)
    if not rows:
        raise SystemExit(f"no per-fold JSON under {base}")
    return rows


def seizure_counts(cfg_manifest: Path | None) -> dict[str, int]:
    """Seizures per EDF, when the manifest is reachable. Optional by design.

    The ranking works without it -- sensitivity and FAR come from the evaluation
    -- but a fold holding three seizures is a better demonstration than one
    holding a single one, and sensitivity alone cannot say which is which.
    """
    if cfg_manifest is None or not cfg_manifest.exists():
        return {}
    import csv

    out: dict[str, int] = {}
    with cfg_manifest.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[r["edf_id"]] = len(json.loads(r.get("seizure_events") or "[]"))
            except (ValueError, KeyError):
                continue
    return out


def test_edf_of(folds_path: Path | None) -> dict[str, list[str]]:
    if folds_path is None or not folds_path.exists():
        return {}
    raw = json.loads(folds_path.read_text(encoding="utf-8"))
    return {f["fold_id"]: sorted(f["test_edf_ids"]) for f in raw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir", type=Path)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="the CHB-MIT manifest CSV; by default "
                         "<artifacts>/manifest/chbmit_manifest.csv")
    ap.add_argument("--folds", type=Path, default=None,
                    help="the splits JSON; by default "
                         "<artifacts>/splits/patient_specific_loso_edf.json")
    ap.add_argument("--split", default="patient_specific_loso_edf",
                    help="which splits file to look for under <artifacts>/splits")
    args = ap.parse_args()

    # Both live under the artifacts directory by construction -- the configs
    # interpolate `${profile.artifacts_dir}` for each. Deriving them beats
    # asking, because a wrong guess degrades silently into a table of "?".
    folds_path = args.folds or (
        args.artifacts_dir / "splits" / f"{args.split}.json")
    manifest_path = args.manifest or (
        args.artifacts_dir / "manifest" / "chbmit_manifest.csv")

    rows = load_rows(args.artifacts_dir, args.bits, args.seed)
    counts = seizure_counts(manifest_path)
    tests = test_edf_of(folds_path)
    for label, path, got in (("splits", folds_path, tests),
                             ("manifest", manifest_path, counts)):
        if not got:
            print(f"note: no {label} at {path}, so that column reads '?'. "
                  f"Pass --{'folds' if label == 'splits' else 'manifest'} "
                  "if it lives elsewhere.")

    for r in rows:
        edfs = tests.get(r["fold_id"], [])
        r["_edfs"] = edfs
        r["_n_sz"] = sum(counts.get(e, 0) for e in edfs) if counts else None

    # Caught everything first; then quietest; then the recording with the most
    # seizures, because a clean sweep of three is worth more than of one.
    def key(r: dict):
        return (-r["sensitivity"], r["far_per_hour"], -(r["_n_sz"] or 0), r["fold_id"])

    rows.sort(key=key)
    perfect = [r for r in rows if r["sensitivity"] >= 1.0]
    silent = [r for r in perfect if r["far_per_hour"] <= 0.0]

    print(f"DFP{args.bits}, seed {args.seed}: {len(rows)} folds scored on the "
          "integer datapath")
    print(f"  {len(perfect)} caught every seizure in their held-out recording")
    print(f"  {len(silent)} of those raised no false alarm at all\n")

    has_n = any(r["_n_sz"] for r in rows)
    head = f"  {'fold':<22}{'sens':>7}{'FAR/h':>9}"
    if has_n:
        head += f"{'seizures':>10}"
    head += "   held-out recording"
    print(head)
    for r in rows[:args.top]:
        line = f"  {r['fold_id']:<22}{r['sensitivity']:>7.3f}{r['far_per_hour']:>9.4f}"
        if has_n:
            line += f"{(r['_n_sz'] if r['_n_sz'] is not None else '-'):>10}"
        line += f"   {', '.join(r['_edfs']) if r['_edfs'] else '?'}"
        print(line)

    if not perfect:
        print("\nNo fold caught every seizure. Pick the highest sensitivity above and "
              "say so\nwhen handing the vectors over -- a demonstration that hides a "
              "miss is worse\nthan one that shows it.")
        return 0

    best = (silent or perfect)[0]
    print(f"\nBest demonstration: {best['fold_id']} -- "
          f"sensitivity {best['sensitivity']:.3f}, FAR {best['far_per_hour']:.4f}/h"
          + (f", {best['_n_sz']} seizure(s)" if best["_n_sz"] else ""))
    if not silent:
        print("  No fold is both perfect and silent, so this one still raises false "
              "alarms.\n  Say that when handing it over.")
    print("\nEach fold carries its own weights under the patient-specific protocol, "
          "so export\nthat fold before taking vectors from it:\n")
    print(f"  python scripts/export_dfp_hardware.py profile=server data=chbmit \\\n"
          f"    model=wearseizure1d_k5only seed={args.seed} train.run_tag=L8 "
          "postprocess=hysteresis_widegrid \\\n"
          f"    +export.fold={best['fold_id']} "
          f"+export.out_dir=hardware/export_{best['fold_id']}\n")
    print(f"  python scripts/export_onset_vectors.py profile=server data=chbmit \\\n"
          f"    model=wearseizure1d_k5only seed={args.seed} train.run_tag=L8 "
          "postprocess=hysteresis_widegrid \\\n"
          f"    +export.model_dir=hardware/export_{best['fold_id']} "
          "+export.pre_s=30 +export.post_s=60\n")
    print("Then check it anywhere, without the dataset:\n")
    print(f"  python scripts/verify_onset_vectors.py "
          f"--model-dir hardware/export_{best['fold_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
