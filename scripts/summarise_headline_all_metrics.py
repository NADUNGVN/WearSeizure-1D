"""Every metric of ONE model over the full 185 h test set, from metrics already on disk.

The headline run (row 43, L8) wrote <fold>.metrics.json for every fold and seed,
including window-level accuracy / sensitivity / specificity / prevalence that no
summary ever printed. This reads them back so the paper can quote event- and
window-level figures from the same model. No training, no inference.

    python scripts/summarise_headline_all_metrics.py \
        "$WEARSEIZURE_ARTIFACTS_DIR/wearseizure1d_k5only/patient_specific_loso_edf/w4s_stride1s__L8"

Event metrics are recomputed the way the paper defines them (macro over 13
patients, micro over seizures, FAR pooled over hours) and printed beside the
logged headline as a check that the right run was read: 0.9489 / 0.9567 / 0.2937.
Window metrics are exposure-weighted over folds (windows are proportional to
hours at a fixed 1 s hop), then averaged over seeds.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

EXPECTED = {"sens_macro": 0.9489, "sens_micro": 0.9567, "far": 0.2937}


def subject(fold_id: str) -> str:
    m = re.match(r"(chb\d+)", fold_id)
    return m.group(1) if m else fold_id.split("_")[0]


def one_seed(folds: list[dict]) -> dict:
    ev = [f["test_event_metrics"] for f in folds]
    per_pt: dict[str, list[int]] = {}
    for f, e in zip(folds, ev):
        p = per_pt.setdefault(subject(f["fold_id"]), [0, 0])
        p[0] += e["n_matched"]
        p[1] += e["n_events"]
    hours = sum(e["exposure_hours"] for e in ev)
    delays = [d for e in ev for d in e["delays_s"]]
    w = [e["exposure_hours"] for e in ev]
    seg = [f["test_segment_metrics"] for f in folds]

    # Runs older than commit 681fddd did not store plain accuracy; it follows
    # exactly from what they did store: acc = sens*prev + spec*(1 - prev).
    for s in seg:
        if "accuracy" not in s:
            p, se, sp = s["prevalence"], s["sensitivity"], s["specificity"]
            s["accuracy"] = sp if p == 0 else se * p + sp * (1 - p)

    def wmean(key: str, weight=lambda s, wi: wi) -> float:
        pairs = [(s[key], weight(s, wi)) for s, wi in zip(seg, w)
                 if key in s and s[key] == s[key]]  # skip NaN (seizure-free folds)
        tot = sum(x for _, x in pairs)
        return sum(v * x for v, x in pairs) / tot if tot else float("nan")

    return {
        "folds": len(folds),
        "hours": hours,
        "sens_macro": st.mean(m / n for m, n in per_pt.values() if n),
        "sens_micro": sum(e["n_matched"] for e in ev) / sum(e["n_events"] for e in ev),
        "far": sum(e["n_false_alarms"] for e in ev) / hours,
        "delay": st.mean(delays) if delays else float("nan"),
        "worst_pt": min(m / n for m, n in per_pt.values() if n),
        # window sensitivity is weighted by ictal windows, specificity by interictal ones
        "sensitivity": wmean("sensitivity", lambda s, wi: wi * s["prevalence"]),
        "specificity": wmean("specificity", lambda s, wi: wi * (1 - s["prevalence"])),
        **{k: wmean(k) for k in ("accuracy", "prevalence", "auroc", "auprc", "f1")},
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).expanduser()
    seeds = sorted(p for p in root.glob("seed*") if p.is_dir())
    if not seeds:
        raise SystemExit(f"no seed*/ directories under {root}")

    rows = []
    for sd in seeds:
        folds = [json.loads(p.read_text()) for p in sorted(sd.glob("*.metrics.json"))]
        if not folds:
            continue
        r = one_seed(folds)
        rows.append(r)
        print(f"{sd.name}: {r['folds']} folds, {r['hours']:.1f} h, sens {r['sens_macro']:.4f}, "
              f"acc {r['accuracy']:.4f}")

    m = {k: st.mean(r[k] for r in rows) for k in rows[0]}
    print(f"\n=== {len(rows)} seeds, {m['folds']:.0f} folds/seed, {m['hours']:.1f} h ===")
    print("EVENT LEVEL")
    for k, label in (("sens_macro", "sensitivity, macro"), ("sens_micro", "sensitivity, micro"), ("far", "FAR /h")):
        print(f"  {label:<22}{m[k]:.4f}   (logged headline {EXPECTED[k]:.4f})")
    print(f"  {'mean delay s':<22}{m['delay']:.2f}")
    print(f"  {'worst patient sens':<22}{m['worst_pt']:.4f}")
    print("WINDOW LEVEL (threshold 0.5, before post-processing)")
    for k in ("accuracy", "sensitivity", "specificity", "auroc", "auprc", "f1"):
        print(f"  {k:<22}{m[k]:.4f}")
    print(f"  {'prevalence':<22}{m['prevalence']:.4f}")
    print(f"  {'never-fires accuracy':<22}{1 - m['prevalence']:.4f}")

    off = [k for k in EXPECTED if abs(m[k] - EXPECTED[k]) > 0.002]
    if off:
        print(f"\nWARNING: {off} differ from the logged headline -- wrong run dir or tag?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
