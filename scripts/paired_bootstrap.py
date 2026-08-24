"""Paired cluster bootstrap between two runs, from the `*.metrics.json` files
`train.py` / `rethreshold.py` already wrote. No training, no GPU, no re-scoring.

Why this exists
---------------
`docs/EXPERIMENT_LOG_G1a.md` currently ranks configurations by comparing point
estimates: 0.9218 (row 22) vs 0.9256 (row 23) vs 0.9359 (row 24). Those are
1.4pp apart on 77 seizures -- about ONE seizure -- and the log says so itself:
row 24 cannot be asserted to beat row 22. Whether the paper can make its
central claim (detection equivalent-or-better than the reproduced baselines, at
roughly 4x lower compute) depends on an interval around that difference, not on
which point estimate happens to be larger.

Usage
-----
    python scripts/paired_bootstrap.py A_DIR B_DIR [--metric sensitivity_macro]

Each DIR is either a single run directory containing `<fold_id>.metrics.json`,
or the directory above it holding `seed0/`, `seed1/`, ... -- in which case the
per-patient counts are averaged over seeds first, so a multi-seed run is
compared as one configuration rather than seed by seed.

Both sides must cover the same patients; the script refuses otherwise, because
a difference computed over two different cohorts is not a difference.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wearseizure.data.splits import subject_from_fold_id
from wearseizure.eval.bootstrap import paired_cluster_bootstrap

# Which way is "better" for each metric. Without this the verdict line reads the
# sign of the difference literally and reports a genuine improvement as a loss,
# which is exactly what happened on the first real Phase 1 result: k5only
# reached FAR 0.2577/h against the default's 0.3658/h, with an interval that
# excludes zero, and the script printed "B better".
METRIC_DIRECTION = {
    "sensitivity_macro": "higher_is_better",
    "sensitivity_micro": "higher_is_better",
    "far_per_hour_micro": "lower_is_better",
    "delay_mean_s": "lower_is_better",
}
METRICS = tuple(METRIC_DIRECTION)


def favours_a(metric: str, delta: float, ci_contains_zero: bool) -> str:
    """Which configuration a difference favours: "A", "B", or "neither"."""
    if ci_contains_zero:
        return "neither"
    ahead = delta > 0 if METRIC_DIRECTION[metric] == "higher_is_better" else delta < 0
    return "A" if ahead else "B"


# `seed<digits>` exactly. A plain `seed*` glob also matches `seed0_noL1`, the
# directory the Phase 2 scripts move an un-pre-trained run into before writing
# the pre-trained one -- and that folded a no-L1 run into the average as a
# fourth "seed", dragging baseline_frontiers2d from 0.9726 down to 0.9591 and
# silently invalidating both architecture comparisons.
_SEED_DIR = re.compile(r"^seed\d+$")


def _seed_dirs(root: Path) -> list[Path]:
    seeds = sorted(p for p in root.glob("seed*") if p.is_dir() and _SEED_DIR.match(p.name))
    return seeds or [root]


def _load_per_patient(root: Path, strategy: str) -> dict[str, dict]:
    """Per-patient totals, averaged across whatever seeds `root` contains.

    Averaging counts (rather than pooling them) keeps `sensitivity` on its
    natural 0-1 scale regardless of how many seeds ran, so a 1-seed run and a
    3-seed run of the same configuration stay directly comparable.
    """
    dirs = _seed_dirs(root)
    per_seed: list[dict[str, dict]] = []
    for d in dirs:
        files = sorted(d.glob("*.metrics.json"))
        if not files:
            raise FileNotFoundError(f"no *.metrics.json under {d}")
        totals: dict[str, dict] = {}
        for f in files:
            payload = json.loads(f.read_text(encoding="utf-8"))
            subject = subject_from_fold_id(payload["fold_id"], strategy)
            m = payload["test_event_metrics"]
            t = totals.setdefault(
                subject,
                {"n_events": 0.0, "n_matched": 0.0, "n_false_alarms": 0.0,
                 "exposure_hours": 0.0, "delays_s": []},
            )
            t["n_events"] += m["n_events"]
            t["n_matched"] += m["n_matched"]
            t["n_false_alarms"] += m["n_false_alarms"]
            t["exposure_hours"] += m["exposure_hours"]
            t["delays_s"].extend(m["delays_s"])
        per_seed.append(totals)

    subjects = set(per_seed[0])
    for t in per_seed[1:]:
        if set(t) != subjects:
            raise ValueError(f"seed directories under {root} cover different patients")

    averaged: dict[str, dict] = {}
    for subject in sorted(subjects):
        rows = [t[subject] for t in per_seed]
        averaged[subject] = {
            "n_events": float(np.mean([r["n_events"] for r in rows])),
            "n_matched": float(np.mean([r["n_matched"] for r in rows])),
            "n_false_alarms": float(np.mean([r["n_false_alarms"] for r in rows])),
            "exposure_hours": float(np.mean([r["exposure_hours"] for r in rows])),
            "delays_s": [d for r in rows for d in r["delays_s"]],
            "n_seeds": len(rows),
        }
    return averaged


def statistic(per_patient: dict[str, dict], ids: Sequence[str], metric: str) -> float:
    rows = [per_patient[i] for i in ids]
    if metric == "sensitivity_macro":
        vals = [r["n_matched"] / r["n_events"] for r in rows if r["n_events"] > 0]
        return float(np.mean(vals)) if vals else float("nan")
    if metric == "sensitivity_micro":
        events = sum(r["n_events"] for r in rows)
        return float(sum(r["n_matched"] for r in rows) / events) if events > 0 else float("nan")
    if metric == "far_per_hour_micro":
        hours = sum(r["exposure_hours"] for r in rows)
        return float(sum(r["n_false_alarms"] for r in rows) / hours) if hours > 0 else float("nan")
    if metric == "delay_mean_s":
        delays = [d for r in rows for d in r["delays_s"]]
        return float(np.mean(delays)) if delays else float("nan")
    raise ValueError(f"unknown metric {metric!r}, expected one of {METRICS}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a_dir", type=Path, help="run directory for configuration A")
    ap.add_argument("b_dir", type=Path, help="run directory for configuration B (the reference)")
    ap.add_argument("--metric", default="sensitivity_macro", choices=METRICS)
    ap.add_argument("--all-metrics", action="store_true", help="report every metric, not just one")
    ap.add_argument("--strategy", default="patient_specific_loso_edf")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed, not a training seed")
    ap.add_argument("--json", type=Path, default=None, help="also write the result here")
    args = ap.parse_args()

    a = _load_per_patient(args.a_dir, args.strategy)
    b = _load_per_patient(args.b_dir, args.strategy)
    if set(a) != set(b):
        raise SystemExit(
            f"A covers {sorted(a)} but B covers {sorted(b)} -- a difference between "
            "two different cohorts is not a difference. Re-run the missing folds first."
        )
    patients = sorted(a)

    metrics = METRICS if args.all_metrics else (args.metric,)
    out = {
        "a_dir": str(args.a_dir),
        "b_dir": str(args.b_dir),
        "n_patients": len(patients),
        "n_seeds_a": a[patients[0]]["n_seeds"],
        "n_seeds_b": b[patients[0]]["n_seeds"],
        "n_boot": args.n_boot,
        "results": {},
    }
    print(f"A = {args.a_dir}  ({out['n_seeds_a']} seed(s))")
    print(f"B = {args.b_dir}  ({out['n_seeds_b']} seed(s))")
    print(f"{len(patients)} patients, {args.n_boot} bootstrap replicates, cluster = patient\n")

    for metric in metrics:
        result = paired_cluster_bootstrap(
            patients,
            lambda ids, m=metric: statistic(a, ids, m) - statistic(b, ids, m),
            n_boot=args.n_boot,
            alpha=args.alpha,
            rng=np.random.default_rng(args.seed),
        )
        result["a"] = statistic(a, patients, metric)
        result["b"] = statistic(b, patients, metric)
        contains_zero = result["ci_low"] <= 0 <= result["ci_high"]
        result["ci_contains_zero"] = bool(contains_zero)
        out["results"][metric] = result
        direction = METRIC_DIRECTION[metric]
        result["direction"] = direction
        result["favours"] = favours_a(metric, result["delta"], contains_zero)
        verdict = {"neither": "indistinguishable", "A": "A better", "B": "B better"}[result["favours"]]
        arrow = "higher better" if direction == "higher_is_better" else "lower better"
        print(
            f"{metric:22s} ({arrow:13s}) A={result['a']:.4f}  B={result['b']:.4f}  "
            f"delta={result['delta']:+.4f}  "
            f"{100 * (1 - args.alpha):.0f}% CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
            f"  -> {verdict}"
        )
    print(
        "\n'delta' is always A minus B. The verdict already accounts for direction, so a "
        "negative delta on a lower-is-better metric reads as 'A better'."
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"written to {args.json}")


if __name__ == "__main__":
    main()
