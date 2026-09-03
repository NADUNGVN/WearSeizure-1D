"""How much longer? Measured from this machine's own file timestamps.

Per-fold cost is not guessable from the outside: it depends on the patient's
recording hours, the architecture, whether a teacher has to be trained first,
and how many runs share the GPU. So this reads the modification times of the
`*.metrics.json` files already on disk, derives the observed seconds-per-fold
for each combination, and projects the rest from that.

Pre-training is counted separately. A NEW architecture (the Phase 7 ladder) has
no cohort initialisations, and building 13 of them per seed is the largest
single cost in that phase -- projecting only from fold times would understate
it badly.

    python scripts/estimate_remaining.py

Read-only.
"""
from __future__ import annotations

import itertools
import os
import statistics
import sys
import time
from pathlib import Path

SPLIT = "patient_specific_loso_edf"
WINDOW = "w4s_stride1s"
FOLDS = 66
INITS = 13

# (label, model, tag) for everything a phase is expected to produce.
PHASES = {
    "Phase 5 -- L3 (multi-channel teacher)": [
        (m, "L3") for m in ("wearseizure1d_k5only", "baseline_frontiers2d")
    ],
    "Phase 5 -- L3single (control arm)": [
        (m, "L3single") for m in ("wearseizure1d_k5only", "baseline_frontiers2d")
    ],
    "Phase 6 -- L8": [("wearseizure1d_k5only", "L8")],
    "Phase 7 -- capacity ladder": [
        (m, "") for m in ("wearseizure1d_k5only_ctx16", "wearseizure1d_k5only_wide")
    ],
}
SEEDS = (0, 1, 2)


def mtimes(d: Path, pattern: str) -> list[float]:
    return sorted(f.stat().st_mtime for f in d.glob(pattern)) if d.is_dir() else []


def rate_from(ts: list[float]) -> float | None:
    """Median seconds between consecutive artifacts.

    Median, not mean: a run that was paused, queued behind another, or resumed
    days later leaves one enormous gap that would swamp an average and make the
    estimate meaningless.
    """
    if len(ts) < 3:
        return None
    gaps = [b - a for a, b in itertools.pairwise(ts) if 0 < b - a < 6 * 3600]
    return statistics.median(gaps) if len(gaps) >= 2 else None


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 36 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def main() -> int:
    art = Path(os.environ.get("WEARSEIZURE_ARTIFACTS_DIR", ""))
    if not art.is_dir():
        print("set WEARSEIZURE_ARTIFACTS_DIR first")
        return 2

    now = time.time()
    all_rates: list[float] = []
    total_remaining = 0.0
    unknown: list[str] = []

    for phase, combos in PHASES.items():
        print(f"\n=== {phase} ===")
        phase_remaining = 0.0
        for model, tag in combos:
            for seed in SEEDS:
                d = art / model / SPLIT / (WINDOW + (f"__{tag}" if tag else "")) / f"seed{seed}"
                ts = mtimes(d, "*.metrics.json")
                done = len(ts)
                left = FOLDS - done
                rate = rate_from(ts)
                if rate:
                    all_rates.append(rate)
                label = f"{model.replace('wearseizure1d_', '').replace('baseline_', '')}/{tag or 'control'}/seed{seed}"
                if left <= 0:
                    print(f"  done   {label:<42} {done}/{FOLDS}")
                    continue
                if rate is None:
                    print(f"  ?      {label:<42} {done}/{FOLDS}  (too few folds to time)")
                    unknown.append(label)
                    continue
                eta = left * rate
                phase_remaining += eta
                live = " <- running now" if done and now - ts[-1] < 1800 else ""
                print(f"  todo   {label:<42} {done}/{FOLDS}  {human(rate)}/fold  ~{human(eta)}{live}")

        # Cohort initialisations, which only a new architecture has to build.
        for model, tag in combos:
            if tag:
                continue
            for seed in SEEDS:
                pre = art / "pretrain" / model / WINDOW / f"seed{seed}"
                have = len(mtimes(pre, "*.pt"))
                if have < INITS:
                    # Priced once at the end, from whichever cache has timings.
                    print(f"  pretrain {model.replace('wearseizure1d_', ''):<40} seed{seed}: {have}/{INITS} inits")
        print(f"  --> phase remaining: {human(phase_remaining)}")
        total_remaining += phase_remaining

    # Pre-training cost, measured from whichever cache already exists.
    pre_rates = []
    for pre in (art / "pretrain").glob("*/" + WINDOW + "/seed*"):
        r = rate_from(mtimes(pre, "*.pt"))
        if r:
            pre_rates.append(r)
    missing_inits = 0
    for combos in PHASES.values():
        for model, tag in combos:
            if tag:
                continue
            for seed in SEEDS:
                have = len(mtimes(art / "pretrain" / model / WINDOW / f"seed{seed}", "*.pt"))
                missing_inits += max(0, INITS - have)
    if missing_inits and pre_rates:
        pre_cost = missing_inits * statistics.median(pre_rates)
        print(f"\ncohort initialisations still to build: {missing_inits} x "
              f"{human(statistics.median(pre_rates))} = ~{human(pre_cost)}")
        total_remaining += pre_cost
    elif missing_inits:
        print(f"\ncohort initialisations still to build: {missing_inits} (no timing data yet)")

    print(f"\nTOTAL REMAINING (if run one after another): ~{human(total_remaining)}")
    if all_rates:
        print(f"observed per-fold time across everything: median {human(statistics.median(all_rates))}, "
              f"range {human(min(all_rates))}-{human(max(all_rates))}")
    print("Two runs in parallel finish sooner than this sum; the GPU is not the bottleneck.")
    if unknown:
        print(f"no estimate yet for: {', '.join(unknown)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
