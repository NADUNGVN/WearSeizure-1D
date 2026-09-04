"""Read the precision sweep and report the loss each format costs.

    python scripts/summarise_precision_sweep.py <artifacts_dir> [--markdown]

Reports delta against FP32 in percentage points, per cell, and says which cells
clear the project's gates. Absolute numbers are shown too, but the delta is the
answer: the question is what quantisation costs, not what the model scores.

The selection rule is fixed in docs/PLAN_quantisation.md and applied here rather
than left to judgement after the fact: take the CHEAPEST format still within
0.5pp of FP32, cheapest meaning dfp8 < int8 < dfp16 < int16 < fp32.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Cheapest first. Cheapness is a hardware statement: fewer bits is less memory
# and a narrower multiplier, and a power-of-two scale removes the requantisation
# multiply entirely.
ORDER = ["dfp8", "int8", "dfp16", "int16", "fp32"]

TARGET_PP, MINIMUM_PP = 0.5, 1.0


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

    if args.markdown:
        print("| format | folds | seeds | event sens | Δ vs FP32 (pp) | FAR/h | Δ FAR/h | verdict |")
        print("|---|--:|--:|--:|--:|--:|--:|---|")
    else:
        print(f"{'format':<8}{'folds':>7}{'seeds':>7}{'sens':>9}{'d_pp':>9}{'FAR/h':>9}{'d_FAR':>9}  verdict")

    for r in rows:
        loss = -r["d_sens_pp"]                      # positive = worse than FP32
        if r["cell"] == "fp32":
            verdict = "reference"
        elif loss <= TARGET_PP:
            verdict = f"within target ({TARGET_PP}pp)"
        elif loss <= MINIMUM_PP:
            verdict = f"within minimum ({MINIMUM_PP}pp), NOT target"
        else:
            verdict = "FAILS both gates"
        if args.markdown:
            print(f"| `{r['cell']}` | {r['n']} | {r['seeds']} | {r['sens']:.4f} | "
                  f"{r['d_sens_pp']:+.2f} | {r['far']:.4f} | {r['d_far']:+.4f} | {verdict} |")
        else:
            print(f"{r['cell']:<8}{r['n']:>7}{r['seeds']:>7}{r['sens']:>9.4f}"
                  f"{r['d_sens_pp']:>+9.2f}{r['far']:>9.4f}{r['d_far']:>+9.4f}  {verdict}")

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

    chosen = next((r["cell"] for r in rows
                   if r["cell"] != "fp32" and -r["d_sens_pp"] <= TARGET_PP), None)
    print()
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
