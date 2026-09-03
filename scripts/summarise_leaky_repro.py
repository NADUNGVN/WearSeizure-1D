"""Turn the per-fold leaky-reproduction JSONs into the Figure 1 table.

    python scripts/summarise_leaky_repro.py <artifacts_dir> [--markdown]

Reports, per rung and model, the mean window-level sensitivity and specificity
across folds -- and beside them the fraction of test windows that share samples
with a training window, which is the number that explains the first row.

Accuracy is printed next to ictal prevalence, always. At 0.6% prevalence a model
that never predicts a seizure scores 99.4%, so the pair is the only thing that
makes the first number readable.
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

# The full table is 66 folds; the stride sweep runs 13, one per patient, by
# design. Anything else is a partial run and gets called out.
EXPECTED_FOLD_COUNTS = (66, 13)


def pretty_stride(tag: str) -> str:
    """stride0p25s -> 0.25s, and the implicit 1s of the untagged layout."""
    return tag.replace("stride", "").replace("p", ".")


def mean(vals: list[float]) -> float:
    # NaN is dropped, not propagated: a fold whose test partition holds one class
    # only has no defined sensitivity, and letting that erase the whole column
    # would hide the folds that do.
    kept = [v for v in vals if not math.isnan(v)]
    return statistics.mean(kept) if kept else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    root = Path(args.artifacts_dir) / "leaky_repro"
    if not root.is_dir():
        print(f"no leaky_repro directory under {args.artifacts_dir}")
        return 1

    # Two layouts: <rung>/<model>/<fold>.json at this project's own 1s stride,
    # and <rung>/<model>/<stride>/<fold>.json for the sweep. The stride is part
    # of the protocol being reproduced -- Chung 2024 slides by ONE SAMPLE -- so
    # it belongs in the row label rather than folded away.
    rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        parent = path.parent.name
        stride = parent if parent.startswith("stride") else "stride1s"
        rows[(d["rung"], d["model"], stride)].append(d)

    if not rows:
        print(f"no per-fold JSON under {root}")
        return 1

    if args.markdown:
        print("| rung | model | stride | folds | window sens | window spec | accuracy "
              "| balanced acc | ictal prevalence | test/train overlap |")
        print("|---|---|---|--:|--:|--:|--:|--:|--:|--:|")
    else:
        print(f"{'rung':<22}{'model':<25}{'stride':>9}{'n':>4}"
              f"{'sens':>9}{'spec':>9}{'acc':>9}{'bal_acc':>9}{'prev':>8}{'overlap':>9}")

    for rung in RUNG_ORDER + sorted({k[0] for k in rows} - set(RUNG_ORDER)):
        for (r, model, stride), ds in sorted(rows.items()):
            if r != rung:
                continue
            sens = mean([d["segment"]["sensitivity"] for d in ds])
            spec = mean([d["segment"]["specificity"] for d in ds])
            bal = mean([d["segment"]["balanced_accuracy"] for d in ds])
            acc = mean([d["segment"].get("accuracy", float("nan")) for d in ds])
            prev = mean([d["segment"]["prevalence"] for d in ds])
            ov = mean([d["test_window_overlap_fraction"] for d in ds])
            st = pretty_stride(stride)
            if args.markdown:
                print(f"| {r} | {model} | {st} | {len(ds)} | {sens:.4f} | {spec:.4f} | "
                      f"{acc:.4f} | {bal:.4f} | {prev:.2%} | {ov:.1%} |")
            else:
                print(f"{r:<22}{model:<25}{st:>9}{len(ds):>4}"
                      f"{sens:>9.4f}{spec:>9.4f}{acc:>9.4f}{bal:>9.4f}{prev:>8.2%}{ov:>9.1%}")

    incomplete = [(k, len(v)) for k, v in sorted(rows.items())
                  if len(v) not in EXPECTED_FOLD_COUNTS]
    if incomplete:
        print("\nPARTIAL -- do not report these as results:")
        for (r, m, st), n in incomplete:
            print(f"  {r} / {m} / stride {pretty_stride(st)}: {n} folds")

    if len({k[2] for k in rows}) > 1:
        print("\nRows differ in FOLD COUNT as well as stride: the 1s rows are the full 66,")
        print("the sweep rows are 13, one per patient. A small difference between a")
        print("66-fold row and a 13-fold row is not evidence of a stride effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
