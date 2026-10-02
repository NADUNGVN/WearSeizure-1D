"""Explain the 0.9495 / 0.9848 gap in the headline sensitivity.

    python scripts/diagnose_headline_gap.py <artifacts_dir>

Two numbers have been quoted for FP32 event sensitivity on the same cohort:
**0.9495** over 66 folds x 3 seeds, and **0.9848** on the folds matched against
the DFP8 export. The brief forbids the second and calls the gap unexplained.
It is worth three and a half seizures, so leaving it unexplained is not an
option for a submission.

Both numbers are read from the same files -- `precision_sweep/fp32/**/*.json`,
written once per (seed, fold). `summarise_precision_sweep.py` averages folds
within a seed and then averages seeds; `summarise_dfp_eval.py` averages the
folds that have a DFP8 counterpart, which in practice is seed 0 alone. So the
gap cannot be a difference of checkpoint, threshold or test exposure, and must
be one of exactly three things:

1. **Seed variance** -- seed 0 really is that much better than seeds 1 and 2.
   Then nothing is broken, the three-seed figure is the right one, and the brief
   should say "seed variance" instead of "unexplained".
2. **An unequal fold set** -- the seeds do not cover the same folds, so the
   means are over different cohorts and are not comparable at all.
3. **A damaged seed** -- one seed's folds are degenerate (all-zero sensitivity,
   missing seizures) in a way that drags the mean rather than sampling it.

This script distinguishes them and says which. It reads existing JSON and
trains nothing.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def patient_of(fold_id: str) -> str:
    return fold_id.split("__")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    args = ap.parse_args()

    base = Path(args.artifacts_dir) / "precision_sweep" / "fp32"
    if not base.is_dir():
        print(f"no FP32 arm under {base}")
        return 1

    rows: dict[int, dict[str, dict]] = defaultdict(dict)
    for path in sorted(base.rglob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rows[d["seed"]][d["fold_id"]] = d["event"]

    if not rows:
        print(f"no per-fold JSON under {base}")
        return 1

    seeds = sorted(rows)
    print(f"FP32 arm: {sum(len(v) for v in rows.values())} fold-results "
          f"over seeds {seeds}\n")

    # 1. Per seed, the two things that could differ: how many folds, and how
    #    well they scored.
    print("  seed   folds   event sens    FAR/h   folds with sens == 0")
    per_seed_sens = {}
    for seed in seeds:
        sens = [r["sensitivity"] for r in rows[seed].values()]
        far = [r["far_per_hour"] for r in rows[seed].values()]
        zeros = sum(1 for s in sens if s == 0.0)
        per_seed_sens[seed] = statistics.mean(sens)
        print(f"  {seed:>4}   {len(sens):>5}   {statistics.mean(sens):>9.4f}"
              f"   {statistics.mean(far):>6.4f}   {zeros:>4}/{len(sens)}")

    macro = statistics.mean(per_seed_sens.values())
    print(f"\n  mean over seeds: {macro:.4f}"
          f"   (the headline, reported as {macro * 100:.2f} %)")
    if 0 in per_seed_sens:
        print(f"  seed 0 alone:    {per_seed_sens[0]:.4f}"
              f"   (what the DFP8 comparison reports)")
        print(f"  gap:             {(per_seed_sens[0] - macro) * 100:+.2f} pp")

    # 2. Do the seeds cover the same folds? If not, the means are over
    #    different cohorts and the comparison was never valid.
    fold_sets = {seed: set(rows[seed]) for seed in seeds}
    common = set.intersection(*fold_sets.values())
    union = set.union(*fold_sets.values())
    print(f"\n  fold coverage: {len(common)} folds in every seed, "
          f"{len(union)} across all seeds")
    unequal = len(common) != len(union)
    if unequal:
        for seed in seeds:
            missing = sorted(union - fold_sets[seed])
            if missing:
                print(f"    seed {seed} is missing {len(missing)}: "
                      f"{', '.join(missing[:5])}"
                      f"{' ...' if len(missing) > 5 else ''}")
        # Re-read the headline on the folds every seed actually has.
        restricted = {
            seed: statistics.mean(rows[seed][f]["sensitivity"] for f in common)
            for seed in seeds
        }
        print("\n  on the common folds only:")
        for seed in seeds:
            print(f"    seed {seed}: {restricted[seed]:.4f}")
        print(f"    mean over seeds: {statistics.mean(restricted.values()):.4f}")

    # 3. Where the seeds disagree, patient by patient. A seed that is merely
    #    unlucky scatters; a seed that is broken collapses on whole patients.
    print("\n  per patient, event sensitivity by seed:")
    patients = sorted({patient_of(f) for f in union})
    header = "    patient " + "".join(f"  seed {s}" for s in seeds) + "   spread"
    print(header)
    worst: list[tuple[float, str]] = []
    for patient in patients:
        cells = []
        for seed in seeds:
            vals = [r["sensitivity"] for f, r in rows[seed].items()
                    if patient_of(f) == patient]
            cells.append(statistics.mean(vals) if vals else float("nan"))
        finite = [c for c in cells if c == c]
        spread = (max(finite) - min(finite)) if len(finite) > 1 else 0.0
        worst.append((spread, patient))
        print(f"    {patient:<8}" + "".join(f"  {c:>6.3f}" for c in cells)
              + f"   {spread:>6.3f}")

    worst.sort(reverse=True)
    top = worst[:3]
    print(f"\n  largest per-patient spread across seeds: "
          + ", ".join(f"{p} {s:.3f}" for s, p in top))

    # Verdict.
    print()
    if unequal:
        print("VERDICT: the seeds do not cover the same folds. The two headline "
              "numbers are\n  means over different cohorts, so the gap is an "
              "artefact of coverage and not a\n  property of the model. Use the "
              "common-fold figure above, and finish or discard\n  the partial "
              "seeds before quoting anything.")
    elif any(len(rows[s]) and statistics.mean(
            r["sensitivity"] for r in rows[s].values()) == 0.0 for s in seeds):
        print("VERDICT: a seed scored zero throughout. That is a broken run, not "
              "variance.")
    else:
        spread_pp = (max(per_seed_sens.values()) - min(per_seed_sens.values())) * 100
        print(f"VERDICT: every seed covers the same {len(common)} folds and none "
              f"collapsed.\n  The spread across seeds is {spread_pp:.2f} pp, and "
              "the gap between the three-seed mean\n  and seed 0 is a consequence "
              "of that spread -- seed variance, not a defect. The\n  three-seed "
              "figure is the one to report, and the brief should say so rather\n"
              "  than calling the gap unexplained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
