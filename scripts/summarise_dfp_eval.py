"""What the real integer datapath costs, against FP32 on the same folds.

    python scripts/summarise_dfp_eval.py <artifacts_dir> [--bits 8] [--markdown]

This reads `dfp_eval/dfp<bits>/seed*/`, written by `export_dfp_hardware.py
+export.evaluate=true`, and compares it against the FP32 arm of the precision
sweep. Unlike that sweep, these numbers come from the datapath the hardware
will actually run: 48-bit accumulator, requantisation by arithmetic shift with
round-to-nearest-ties-away, saturation to the data width, and the six extra
quantisation points the hardware introduces between each depthwise and
pointwise pair. The sweep models none of those, so where the two disagree this
one is the answer.

The comparison is paired and clustered by patient, for the reason every
comparison in this project is: the two arms are the same model on the same
folds differing only in numeric format, and folds from one patient are not
independent of one another.

The reported loss is the WORST CASE THE DATA STILL ALLOWS -- the lower end of
the interval -- not the point estimate. Event sensitivity is a step function of
the scores, so a point estimate here is worth about a tenth of one seizure, and
ranking formats by it is more precision than the measurement carries.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET_PP, MINIMUM_PP = 0.5, 1.0
N_BOOT, N_EVENTS = 10_000, 77


def load_dfp(root: Path, bits: int) -> dict[tuple[int, str], dict]:
    out = {}
    # Both scoring arms: dfp<bits> holds the requantised-logit run and
    # dfp<bits>_acc the raw-accumulator one. Which is which comes from each
    # row's own `raw_margin` field, not from the directory name.
    for arm in sorted((root / "dfp_eval").glob(f"dfp{bits}*")):
        for path in sorted(arm.rglob("*.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            out[(bool(d.get("raw_margin", False)), d["seed"], d["fold_id"])] = d
    return out


def load_fp32(root: Path) -> dict[tuple[int, str], dict]:
    """The FP32 arm of the precision sweep, which is the reference here.

    Read from the sweep rather than recomputed: it used the same frozen
    thresholds, so the two arms differ only in numeric format.
    """
    out = {}
    base = root / "precision_sweep" / "fp32"
    for path in sorted(base.rglob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        out[(d["seed"], d["fold_id"])] = {
            "sensitivity": d["event"]["sensitivity"],
            "far_per_hour": d["event"]["far_per_hour"],
        }
    return out


def paired_delta(pairs: dict[str, list[tuple[float, float]]]) -> tuple[float, float, float]:
    patients = sorted(pairs)
    if not patients:
        return float("nan"), float("nan"), float("nan")

    def delta(sample: list[str]) -> float:
        a = [x for p in sample for x, _ in pairs[p]]
        b = [y for p in sample for _, y in pairs[p]]
        return (statistics.mean(a) - statistics.mean(b)) * 100

    rng = np.random.default_rng(0)
    boots = [delta([patients[i] for i in rng.integers(0, len(patients), len(patients))])
             for _ in range(N_BOOT)]
    return delta(patients), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    root = Path(args.artifacts_dir)
    dfp = load_dfp(root, args.bits)
    if not dfp:
        print(f"no dfp_eval/dfp{args.bits}* results under {root}")
        return 1
    fp32 = load_fp32(root)

    # Split by whether the score came from the requantised logits or the raw
    # accumulator: they answer different questions and averaging them would
    # answer neither.
    arms: dict[bool, dict] = defaultdict(dict)
    for (raw, seed, fold_id), row in dfp.items():
        arms[raw][(seed, fold_id)] = row

    print(f"DFP{args.bits} through the integer datapath")
    if not fp32:
        print(f"  no FP32 arm under {root / 'precision_sweep' / 'fp32'}; reporting "
              "absolute numbers only.\n  Run the precision sweep first to get a delta.")

    for raw, rows in sorted(arms.items()):
        label = "48-bit accumulator" if raw else "requantised logits"
        sens = [r["sensitivity"] for r in rows.values()]
        far = [r["far_per_hour"] for r in rows.values()]
        seeds = sorted({k[0] for k in rows})
        print(f"\n  scored from the {label}  ({len(rows)} folds, seeds {seeds})")
        print(f"    event sensitivity {statistics.mean(sens):.4f}"
              f"    FAR/h {statistics.mean(far):.4f}")

        matched = {k: v for k, v in rows.items() if k in fp32}
        if not matched:
            continue
        if len(matched) < len(rows):
            print(f"    NOTE: only {len(matched)} of {len(rows)} folds have an FP32 "
                  "counterpart; the delta below covers those.")

        pairs_s: dict[str, list[tuple[float, float]]] = defaultdict(list)
        pairs_f: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for key, row in matched.items():
            patient = key[1].split("__")[0]
            pairs_s[patient].append((row["sensitivity"], fp32[key]["sensitivity"]))
            pairs_f[patient].append((row["far_per_hour"], fp32[key]["far_per_hour"]))

        d, lo, hi = paired_delta(pairs_s)
        df, flo, fhi = paired_delta(pairs_f)
        worst = -lo
        gate = ("within the 0.5pp target" if worst <= TARGET_PP
                else "within the 1.0pp minimum" if worst <= MINIMUM_PP
                else "EXCEEDS both gates")
        print(f"    vs FP32: {d:+.2f} pp  95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"   = {d / 100 * N_EVENTS:+.2f} of one seizure")
        print(f"    loses up to {worst:.2f} pp -- {gate}")
        print(f"    FAR/h delta {df / 100:+.4f}")
        if lo <= 0 <= hi:
            print("    The interval spans zero: on this cohort this format is not "
                  "measurably\n    different from FP32. What separates formats is the "
                  "WIDTH of the interval,\n    not the point estimate.")

    if len(arms) == 2:
        a = statistics.mean([r["sensitivity"] for r in arms[False].values()])
        b = statistics.mean([r["sensitivity"] for r in arms[True].values()])
        print(f"\n  reading the accumulator instead of the logits: {100 * (b - a):+.2f} pp")
        print("  That is the price of pushing the final layer through the same "
              "requantiser as\n  every other one. The value is already in the PE, so "
              "recovering it costs no\n  hardware -- only an instruction bit.")
    else:
        print("\n  Only one scoring arm present. Run the other with "
              "+export.raw_margin=true\n  to price the final requantisation "
              "(section 3.3 of MODEL_TEAM_TASKS.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
