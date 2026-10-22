"""Lever L9: did onset-proximity sampling actually make the detector faster?

    python scripts/summarise_onset_weight.py <artifacts_dir> --baseline L8 --arm L9

Compares two run tags fold for fold on the three things that decide whether the
lever is worth keeping, in this order:

1. **Detection delay.** The point of the lever. Measured from each matched
   event's `delays_s`, which the evaluator already records per fold.
2. **Event sensitivity.** The lever emphasises the hardest, most
   pre-ictal-looking windows in the set, so it could plausibly teach the model
   to fire on signal that is not a seizure yet -- costing sensitivity nowhere
   and FAR everywhere, or trading one for the other.
3. **False-alarm rate.** The same risk from the other side.

**A delay improvement that costs sensitivity or FAR is not an improvement.**
The device is judged on catching seizures with few false alarms; reacting
faster to the ones it still catches is worth something only if it catches as
many. All three are reported with a patient-clustered paired bootstrap, and the
verdict is read off the **worst case the interval allows**, which is this
project's convention everywhere else.

Delays are only defined for events that were matched, so the delay comparison
runs over folds where BOTH arms detected something. That is the honest pairing,
and the script says how many folds it had to drop to get it -- if the baseline
detects events the new arm misses, a faster mean delay is survivorship, not
speed.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10_000


def patient_of(fold_id: str) -> str:
    return fold_id.split("__")[0]


def load_arm(art: Path, model: str, split: str, window: str, tag: str,
             seed: int) -> dict[str, dict]:
    name = f"{window}__{tag}" if tag else window
    root = art / model / split / name / f"seed{seed}"
    rows: dict[str, dict] = {}
    for p in sorted(root.glob("*.metrics.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows[d["fold_id"]] = d["test_event_metrics"]
    return rows


def paired_delta(pairs: dict[str, list[tuple[float, float]]],
                 scale: float = 1.0) -> tuple[float, float, float]:
    """Mean(arm) - mean(baseline), resampling PATIENTS, not folds.

    Folds from one patient share training data and a physiology; treating them
    as independent would narrow every interval in this project by pretending
    there are 66 independent observations where there are 13.
    """
    patients = sorted(pairs)
    if not patients:
        return float("nan"), float("nan"), float("nan")

    def delta(sample: list[str]) -> float:
        a = [x for p in sample for x, _ in pairs[p]]
        b = [y for p in sample for _, y in pairs[p]]
        return (st.mean(a) - st.mean(b)) * scale

    rng = np.random.default_rng(0)
    boots = [delta([patients[i] for i in rng.integers(0, len(patients), len(patients))])
             for _ in range(N_BOOT)]
    return delta(patients), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir", type=Path)
    ap.add_argument("--baseline", default="L8")
    ap.add_argument("--arm", default="L9")
    ap.add_argument("--model", default="wearseizure1d_k5only")
    ap.add_argument("--split", default="patient_specific_loso_edf")
    ap.add_argument("--window", default="w4s_stride1s")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base = load_arm(a.artifacts_dir, a.model, a.split, a.window, a.baseline, a.seed)
    arm = load_arm(a.artifacts_dir, a.model, a.split, a.window, a.arm, a.seed)
    if not base or not arm:
        print(f"baseline {a.baseline}: {len(base)} folds; arm {a.arm}: {len(arm)} folds. "
              "Both must exist.")
        return 1

    common = sorted(set(base) & set(arm))
    print(f"{a.arm} against {a.baseline}, seed {a.seed}: "
          f"{len(common)} folds in both ({len(base)} / {len(arm)} present)")
    if len(common) < max(len(base), len(arm)):
        print(f"  NOTE: comparing the {len(common)} folds both arms have finished. "
              "A partial\n  arm makes every number below provisional.")

    # Sensitivity and FAR over every shared fold.
    sens: dict[str, list[tuple[float, float]]] = defaultdict(list)
    far: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for f in common:
        p = patient_of(f)
        sens[p].append((arm[f]["sensitivity"], base[f]["sensitivity"]))
        far[p].append((arm[f]["far_per_hour"], base[f]["far_per_hour"]))

    # Delay only where both arms matched something, so the mean is not taken
    # over a different set of events in each arm.
    delay: dict[str, list[tuple[float, float]]] = defaultdict(list)
    dropped = 0
    for f in common:
        da, db = arm[f]["delays_s"], base[f]["delays_s"]
        if da and db:
            delay[patient_of(f)].append((st.mean(da), st.mean(db)))
        else:
            dropped += 1

    print(f"\n{'metric':<22}{'baseline':>10}{'arm':>10}{'delta':>10}{'95% CI':>22}")

    def row(name, pairs, scale, unit, lower_is_better):
        if not pairs:
            print(f"  {name:<20}{'--':>10}{'--':>10}{'no paired folds':>42}")
            return None
        b = st.mean([y for v in pairs.values() for _, y in v])
        m = st.mean([x for v in pairs.values() for x, _ in v])
        d, lo, hi = paired_delta(pairs, scale)
        print(f"  {name:<20}{b * scale:>9.2f}{unit}{m * scale:>9.2f}{unit}"
              f"{d:>+9.2f}{unit}  [{lo:+.2f}, {hi:+.2f}]{unit}")
        return d, lo, hi, lower_is_better

    r_delay = row("detection delay", delay, 1.0, "s", True)
    r_sens = row("event sensitivity", sens, 100.0, "pp", False)
    r_far = row("FAR per hour", far, 1.0, "/h", True)
    if dropped:
        print(f"\n  {dropped} fold(s) left out of the delay row: one arm matched no "
              "event there.")

    print(f"\n{'=' * 78}")
    if r_delay is None:
        print("No fold has a delay in both arms. Nothing to conclude.")
        return 0

    d, lo, hi, _ = r_delay
    faster = hi < 0.0           # the whole interval is an improvement
    slower = lo > 0.0
    harms = []
    if r_sens and r_sens[1] < -1.0:
        harms.append(f"sensitivity could be {abs(r_sens[1]):.2f} pp worse")
    if r_far and r_far[2] > 0.05:
        harms.append(f"FAR could be {r_far[2]:+.3f}/h worse")

    if faster and not harms:
        print(f"KEEP THE LEVER. Delay is lower by {abs(d):.2f} s and the interval "
              f"excludes zero\n  ([{lo:+.2f}, {hi:+.2f}] s), with no measurable cost to "
              "sensitivity or FAR.")
    elif faster and harms:
        print(f"FASTER BUT NOT FREE. Delay falls {abs(d):.2f} s, and " +
              "; ".join(harms) + ".\n  The device is judged on catching seizures, so "
              "this is a trade to decide\n  deliberately, not an improvement to adopt.")
    elif slower:
        print(f"REJECT. The arm is SLOWER by {d:.2f} s and the interval excludes zero.")
    else:
        print(f"NOT ESTABLISHED. The delta is {d:+.2f} s but the interval "
              f"[{lo:+.2f}, {hi:+.2f}] spans zero,\n  so the data does not distinguish "
              "the arms. Worst case the interval allows is\n  "
              f"{hi:+.2f} s -- read that, not the point estimate. More seeds would "
              "narrow it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
