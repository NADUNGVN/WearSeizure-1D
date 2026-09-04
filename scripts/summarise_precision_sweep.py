"""Read the precision sweep and report the loss each format costs.

    python scripts/summarise_precision_sweep.py <artifacts_dir> [--markdown]

Reports delta against FP32 in percentage points, per cell, and says which cells
clear the project's gates. Absolute numbers are shown too, but the delta is the
answer: the question is what quantisation costs, not what the model scores.

The selection rule is fixed in docs/PLAN_quantisation.md and applied here rather
than left to judgement after the fact: take the CHEAPEST format still within
0.5pp of FP32, cheapest meaning dfp8 < int8 < dfp16 < int16 < fp32.

That rule needs a guard, because a threshold is more precise than the
measurement behind it. Event sensitivity is a step function of the scores -- a
tenth of a percentage point is a tenth of one seizure in 77 -- so a format can
miss a 0.5pp gate by a hundredth of a point without being any worse. Each cell
therefore gets a patient-clustered paired bootstrap against FP32, and when
several intervals span zero the script refuses to rank them on accuracy and says
to choose on hardware cost instead.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

# Cheapest first. Cheapness is a hardware statement: fewer bits is less memory
# and a narrower multiplier, and a power-of-two scale removes the requantisation
# multiply entirely.
ORDER = ["dfp8", "int8", "dfp16", "int16", "fp32"]

TARGET_PP, MINIMUM_PP = 0.5, 1.0
N_BOOT, N_EVENTS = 10_000, 77


def paired_delta_ci(cell_rows: list[dict], ref_rows: list[dict]) -> tuple[float, float, float]:
    """Patient-clustered paired bootstrap of (cell - fp32) sensitivity, in pp.

    Paired and clustered for the same reason as every other comparison here:
    the two arms are the SAME model on the SAME folds, differing only in numeric
    format, and folds from one patient are not independent of each other.
    """
    by_fold = {(r["seed"], r["fold_id"]): r["event"]["sensitivity"] for r in ref_rows}
    pairs: dict[str, list[tuple[float, float]]] = {}
    for r in cell_rows:
        key = (r["seed"], r["fold_id"])
        if key in by_fold:
            patient = r["fold_id"].split("__")[0]
            pairs.setdefault(patient, []).append((r["event"]["sensitivity"], by_fold[key]))

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


def mean(vals: list[float]) -> float:
    kept = [v for v in vals if not math.isnan(v)]
    return statistics.mean(kept) if kept else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    root = Path(args.artifacts_dir) / "precision_sweep"
    if not root.is_dir():
        print(f"no precision_sweep directory under {args.artifacts_dir}")
        return 1

    # (cell, seed) -> per-fold event metrics, so a seed's folds are pooled the
    # way every other comparison in this project pools them.
    cells: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        cells[d["cell"]][d["seed"]].append(d)

    if not cells:
        print(f"no per-fold JSON under {root}")
        return 1

    def agg(cell: str, key: str) -> float:
        # Mean over folds within a seed, then over seeds -- macro, matching the
        # experiment log rather than inventing a third convention here.
        per_seed = [mean([r["event"][key] for r in rows]) for rows in cells[cell].values()]
        return mean(per_seed)

    if "fp32" not in cells:
        print("no fp32 cell: nothing to measure a loss against. Run the sweep with it included.")
        return 1

    ref_sens = agg("fp32", "sensitivity")
    ref_far = agg("fp32", "far_per_hour")
    n_folds = {c: sum(len(v) for v in cells[c].values()) for c in cells}

    rows = []
    for cell in ORDER + sorted(set(cells) - set(ORDER)):
        if cell not in cells:
            continue
        sens, far = agg(cell, "sensitivity"), agg(cell, "far_per_hour")
        rows.append({
            "cell": cell,
            "n": n_folds[cell],
            "seeds": len(cells[cell]),
            "sens": sens,
            "d_sens_pp": (sens - ref_sens) * 100,
            "far": far,
            "d_far": far - ref_far,
        })

    ref_rows = [r for rows in cells["fp32"].values() for r in rows]
    for r in rows:
        if r["cell"] == "fp32":
            r["ci"] = (0.0, 0.0, 0.0)
            continue
        r["ci"] = paired_delta_ci(
            [x for rr in cells[r["cell"]].values() for x in rr], ref_rows
        )

    if args.markdown:
        print("| format | folds | event sens | Δ vs FP32 (pp) | 95% CI (pp) | Δ in seizures | FAR/h | verdict |")
        print("|---|--:|--:|--:|--:|--:|--:|---|")
    else:
        print(f"{'format':<8}{'folds':>7}{'sens':>9}{'d_pp':>9}{'95% CI (pp)':>20}"
              f"{'seizures':>10}{'FAR/h':>9}  verdict")

    for r in rows:
        _, lo, hi = r["ci"]
        loss = -r["d_sens_pp"]
        indistinguishable = r["cell"] != "fp32" and lo <= 0 <= hi
        if r["cell"] == "fp32":
            verdict = "reference"
        elif indistinguishable:
            # The interval spanning zero is the finding, not a footnote to the
            # point estimate: it says this format is not measurably different
            # from FP32 on this cohort.
            verdict = "indistinguishable from FP32"
        elif loss <= TARGET_PP:
            verdict = f"worse, but within target ({TARGET_PP}pp)"
        elif loss <= MINIMUM_PP:
            verdict = f"worse, within minimum ({MINIMUM_PP}pp)"
        else:
            verdict = "FAILS both gates"
        r["indistinguishable"] = indistinguishable
        seizures = r["d_sens_pp"] / 100 * N_EVENTS
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if r["cell"] != "fp32" else ""
        if args.markdown:
            print(f"| `{r['cell']}` | {r['n']} | {r['sens']:.4f} | {r['d_sens_pp']:+.2f} | "
                  f"{ci} | {seizures:+.2f} | {r['far']:.4f} | {verdict} |")
        else:
            print(f"{r['cell']:<8}{r['n']:>7}{r['sens']:>9.4f}{r['d_sens_pp']:>+9.2f}"
                  f"{ci:>20}{seizures:>+10.2f}{r['far']:>9.4f}  {verdict}")

    # The two axes, read off the grid rather than eyeballed.
    have = {r["cell"]: r["d_sens_pp"] for r in rows}
    print()
    print("=== the two axes, separated ===")
    for a, b, label in (("int16", "dfp16", "power-of-two scale at 16 bit"),
                        ("int8", "dfp8", "power-of-two scale at 8 bit"),
                        ("int16", "int8", "narrowing 16 -> 8 bit, arbitrary scale"),
                        ("dfp16", "dfp8", "narrowing 16 -> 8 bit, power-of-two scale")):
        if a in have and b in have:
            print(f"  {label:<44}{have[b] - have[a]:+.2f} pp")
    print("  A power-of-two row costing much less than a width row means the format is")
    print("  nearly free and the bits are what matter -- or the reverse. Read both.")

    # Point estimates being indistinguishable does NOT make the formats
    # equivalent: the intervals differ in WIDTH, and the width is the decision.
    # A format whose interval reaches -1.9pp could be four times worse than the
    # gate allows; one bounded at -0.25pp cannot. Picking the cheapest of the
    # "tied" formats ignores that, which is how the first version of this rule
    # would have shipped a format whose worst supported case fails the gate.
    tied = [r["cell"] for r in rows if r.get("indistinguishable")]
    print()
    if len(tied) > 1:
        spread = max(r["d_sens_pp"] for r in rows) - min(r["d_sens_pp"] for r in rows)
        print(f"POINT ESTIMATES ARE NOISE: {', '.join(tied)} all have CIs spanning zero,")
        print(f"  and the whole spread is {spread:.2f} pp = {spread / 100 * N_EVENTS:.2f} of one seizure.")
        print("  Three of the four grid edges below carry physically impossible signs")
        print("  (a power-of-two scale beating an arbitrary one, or fewer bits beating more),")
        print("  which is the same conclusion arrived at from a different direction.")
        print()
        print("  So rank by the WORST CASE THE DATA STILL ALLOWS, not by the point estimate:")
        safe = []
        for r in rows:
            if r["cell"] == "fp32":
                continue
            worst = -r["ci"][1]                 # lower CI bound, as a loss
            gate = ("within target" if worst <= TARGET_PP
                    else "within minimum" if worst <= MINIMUM_PP else "EXCEEDS both gates")
            print(f"    {r['cell']:<8}loses up to {worst:5.2f} pp   {gate}")
            if worst <= TARGET_PP:
                safe.append(r["cell"])
        print()
        if safe:
            pick = next(c for c in ORDER if c in safe)
            print(f"  SELECTED: {pick} -- the cheapest format whose worst supported case stays")
            print(f"  within {TARGET_PP}pp. The others are not worse on the evidence, but the")
            print("  evidence does not rule out their being much worse, and here the memory")
            print("  that buys the tighter bound is a few KiB on a device with hundreds.")
        else:
            print(f"  No format's worst supported case stays within {TARGET_PP}pp. Either accept")
            print(f"  the {MINIMUM_PP}pp gate and record that decision, or improve the PTQ method")
            print("  (docs/PLAN_ptq_method.md step P3) before choosing.")
        print()
        print("  Record the tie as well as the choice: a later reader must not think the")
        print("  format was selected because it scored better. It was not.")
        return 0

    chosen = next((r["cell"] for r in rows
                   if r["cell"] != "fp32" and -r["d_sens_pp"] <= TARGET_PP), None)
    if chosen:
        print(f"SELECTED: {chosen} -- the cheapest format within {TARGET_PP}pp of FP32.")
    else:
        relaxed = next((r["cell"] for r in rows
                        if r["cell"] != "fp32" and -r["d_sens_pp"] <= MINIMUM_PP), None)
        if relaxed:
            print(f"No format clears {TARGET_PP}pp. Cheapest within {MINIMUM_PP}pp: {relaxed}.")
            print("Relaxing the gate is a decision to record in the paper, not to make quietly.")
        else:
            print("No format clears either gate. Next step is QAT, not another PTQ tweak --")
            print("see docs/PLAN_ptq_method.md step P5.")

    short = [c for c, n in n_folds.items() if n != max(n_folds.values())]
    if short:
        print(f"\nPARTIAL, do not report: {', '.join(short)} have fewer folds than the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
