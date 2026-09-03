"""Turn the per-fold leaky-reproduction JSONs into the Figure 1 table.

    python scripts/summarise_leaky_repro.py <artifacts_dir> [--markdown]

Reports, per rung and model, the mean window-level sensitivity and specificity
across folds -- and beside them the fraction of test windows that share samples
with a training window, which is the number that explains the first row.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Ordered as the ladder is, so the table reads top to bottom as leaks are removed.
RUNG_ORDER = ["A_as_published", "B_split_by_recording", "C_no_fitting_leak"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    root = Path(args.artifacts_dir) / "leaky_repro"
    if not root.is_dir():
        print(f"no leaky_repro directory under {args.artifacts_dir}")
        return 1

    rows = defaultdict(list)
    for path in sorted(root.glob("*/*/*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rows[(d["rung"], d["model"])].append(d)

    if args.markdown:
        print("| rung | model | folds | window sens | window spec | balanced acc | test/train overlap |")
        print("|---|---|--:|--:|--:|--:|--:|")
    else:
        print(f"{'rung':<24}{'model':<26}{'n':>4}{'sens':>9}{'spec':>9}{'bal_acc':>9}{'overlap':>9}")

    def mean(vals):
        # NaN is dropped, not propagated: a fold whose test partition holds one
        # class only has no defined sensitivity, and letting that erase the
        # whole column would hide the 65 folds that do.
        vals = [v for v in vals if not math.isnan(v)]
        return statistics.mean(vals) if vals else float("nan")

    for rung in RUNG_ORDER + sorted({k[0] for k in rows} - set(RUNG_ORDER)):
        for (r, model), ds in sorted(rows.items()):
            if r != rung:
                continue
            sens = mean([d["segment"]["sensitivity"] for d in ds])
            spec = mean([d["segment"]["specificity"] for d in ds])
            bal = mean([d["segment"]["balanced_accuracy"] for d in ds])
            ov = mean([d["test_window_overlap_fraction"] for d in ds])
            if args.markdown:
                print(f"| `{r}` | `{model}` | {len(ds)} | {sens:.4f} | {spec:.4f} | {bal:.4f} | {ov:.1%} |")
            else:
                print(f"{r:<24}{model:<26}{len(ds):>4}{sens:>9.4f}{spec:>9.4f}{bal:>9.4f}{ov:>9.1%}")

    incomplete = [(k, len(v)) for k, v in sorted(rows.items()) if len(v) != 66]
    if incomplete:
        print("\nNOT 66 folds -- do not report these as results:")
        for (r, m), n in incomplete:
            print(f"  {r} / {m}: {n}/66")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
