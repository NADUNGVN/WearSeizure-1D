"""The 24-case result, with the excluded groups kept apart.

    python scripts/summarise_full_cohort.py <artifacts_dir>

Pooling all 24 cases into one number would be the wrong summary. Chung et al.
excluded ten cases for two different reasons, and only one of them is about the
model:

  chb06, 09, 13, 19   seizure location NOT near any wearable position
  chb12, 14, 16, 18, 20, 21   seizure location could not be identified

For the first group a wearable electrode cannot reach the seizure at all. A
poor result there is anatomy. Reporting a single pooled figure would let that
anatomical limit be read as a modelling failure, which is why this prints the
groups separately and refuses to print a pooled headline.

The comparison to make is against the channel ablation's 1-channel arm --
0.9179 event sensitivity over 66 folds x 3 seeds -- because that arm uses this
same recipe: trained from scratch per fold, no cohort pre-training, no
distillation.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ABLATION_1CH_EVENT_SENS = 0.9179   # docs/EXPERIMENT_LOG_G1a.md section 2n, row 58

GROUP_LABEL = {
    "confirmed": "clinically confirmed (the 13-case cohort)",
    "seizure_not_near_wearable": "seizure NOT near any wearable position",
    "location_unidentified": "seizure location unidentified",
    "other_excluded": "excluded for other reasons",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    args = ap.parse_args()

    root = Path(args.artifacts_dir) / "full_cohort"
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.json"))]
    if not rows:
        print(f"no results under {root}")
        return 1

    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r.get("exclusion_group", "other_excluded")].append(r)

    print(f"{len(rows)} folds, {len({r['subject'] for r in rows})} cases, "
          f"seeds {sorted({r['seed'] for r in rows})}\n")
    print(f"{'group':<42}{'folds':>7}{'cases':>7}{'event sens':>12}{'FAR/h':>9}{'AUROC':>8}")
    for key in ("confirmed", "location_unidentified",
                "seizure_not_near_wearable", "other_excluded"):
        g = by_group.get(key)
        if not g:
            continue
        print(f"{GROUP_LABEL[key]:<42}{len(g):>7}{len({r['subject'] for r in g}):>7}"
              f"{st.mean(r['sensitivity'] for r in g):>12.4f}"
              f"{st.mean(r['far_per_hour'] for r in g):>9.4f}"
              f"{st.mean(r['segment']['auroc'] for r in g):>8.4f}")

    confirmed = by_group.get("confirmed", [])
    if confirmed:
        delta = st.mean(r["sensitivity"] for r in confirmed) - ABLATION_1CH_EVENT_SENS
        print(f"\nthe confirmed cases here vs the channel ablation's 1-channel arm "
              f"({ABLATION_1CH_EVENT_SENS:.4f}): {100 * delta:+.2f} pp")
        print("  Same recipe -- from scratch, no pre-training, no distillation -- so a "
              "large\n  difference would mean the two runs are not comparable and "
              "should be explained\n  before either is quoted.")

    print("\nper case")
    print(f"{'case':<9}{'group':<28}{'folds':>6}{'channel(s)':>26}{'event sens':>12}")
    for subject in sorted({r["subject"] for r in rows}):
        g = [r for r in rows if r["subject"] == subject]
        chans = sorted({r["channel"] for r in g})
        chan_str = ",".join(chans) if len(chans) <= 2 else f"{len(chans)} differing"
        print(f"{subject:<9}{g[0].get('exclusion_group', '?'):<28}{len(g):>6}"
              f"{chan_str:>26}{st.mean(r['sensitivity'] for r in g):>12.4f}")

    # Which position the selector picked, for the cases that had no confirmed
    # one. If it lands on a single position everywhere, the selection is doing
    # nothing and a fixed choice would be simpler and easier to defend.
    unconfirmed = [r for r in rows if not r.get("clinically_confirmed")]
    if unconfirmed:
        picks: dict[str, int] = defaultdict(int)
        for r in unconfirmed:
            picks[r["channel"]] += 1
        print(f"\nselector's choices across {len(unconfirmed)} unconfirmed folds:")
        for ch, n in sorted(picks.items(), key=lambda kv: -kv[1]):
            print(f"  {ch:<10}{n:>4}  ({100 * n / len(unconfirmed):.0f} %)")
        if len(picks) == 1:
            print("  It chose the same position every time, so per-fold selection "
                  "bought nothing\n  here and a fixed position would be simpler to "
                  "defend. Worth saying so.")

    print("\nNo pooled 24-case figure is printed on purpose. The groups answer "
          "different\nquestions, and a single mean would let an anatomical limit read "
          "as a modelling\nfailure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
